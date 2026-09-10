"""CLI entry point: python -m exporter list|run|card"""

import argparse
import sys

from exporter import registry
# Import models to trigger registration
import exporter.models  # noqa: F401


def _fetch_published_repos(org="inference4j"):
    """Fetch the set of repo IDs published under a HuggingFace org."""
    try:
        from huggingface_hub import list_models
        return {m.id for m in list_models(author=org)}
    except Exception:
        return None


def cmd_list(_args):
    """Print a table of all registered models."""
    models = registry.all_models()
    if not models:
        print("No models registered.")
        return

    published = _fetch_published_repos()
    show_status = published is not None

    from exporter.base import MirroredModel

    # Column widths
    name_w = max(len(m.name) for m in models)
    type_w = 8  # "mirrored" or "exported"
    status_w = 10
    source_w = max((len(m.source_repo) for m in models if m.source_repo), default=6)

    header = f"{'Name':<{name_w}}  {'Type':<{type_w}}  "
    sep = f"{'-' * name_w}  {'-' * type_w}  "
    if show_status:
        header += f"{'Published':<{status_w}}  "
        sep += f"{'-' * status_w}  "
    header += f"{'Source':<{source_w}}  Repo ID"
    sep += f"{'-' * source_w}  {'-' * 40}"

    print(header)
    print(sep)
    for m in models:
        model_type = "mirrored" if isinstance(m, MirroredModel) else "exported"
        source = m.source_repo or "—"
        line = f"{m.name:<{name_w}}  {model_type:<{type_w}}  "
        if show_status:
            status = "yes" if m.repo_id in published else "no"
            line += f"{status:<{status_w}}  "
        line += f"{source:<{source_w}}  {m.repo_id}"
        print(line)


def cmd_run(args):
    """Run staging + upload for specified models."""
    from exporter import upload

    if not args.models and not args.all:
        print("Error: specify model names or use --all")
        print(f"Available models: {', '.join(registry.names())}")
        sys.exit(1)

    if args.all:
        models = registry.all_models()
    else:
        models = []
        for name in args.models:
            model = registry.get(name)
            if model is None:
                print(f"Error: unknown model '{name}'")
                print(f"Available models: {', '.join(registry.names())}")
                sys.exit(1)
            models.append(model)

    if args.stage_to:
        from pathlib import Path
        for model in models:
            upload.stage_to(model, Path(args.stage_to), update=args.update)
        return

    print(f"Models to process: {len(models)}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    if args.update:
        print("[UPDATE MODE] Exporting from upstream HEAD and re-pinning the lock")

    for model in models:
        upload.process(model, dry_run=args.dry_run, update=args.update)

    print(f"\nDone! Processed {len(models)} model(s).")


def _select(names):
    """Resolve model names to definitions, or all of them when none are given."""
    if not names:
        return registry.all_models()
    models = []
    for name in names:
        model = registry.get(name)
        if model is None:
            print(f"Error: unknown model '{name}'")
            print(f"Available models: {', '.join(registry.names())}")
            sys.exit(1)
        models.append(model)
    return models


def cmd_check(args):
    """Report which models have drifted from their locked upstream revision."""
    import json as _json

    from exporter import check, lock

    models = _select(args.models)
    results = check.check_all(models, deep=args.deep)

    if args.json:
        print(_json.dumps([r.to_json() for r in results], indent=2))
    else:
        name_w = max(len(r.name) for r in results)
        status_w = max(len(r.status) for r in results)
        print(f"{'Model':<{name_w}}  {'Status':<{status_w}}  Detail")
        print(f"{'-' * name_w}  {'-' * status_w}  {'-' * 50}")
        for r in results:
            print(f"{r.name:<{name_w}}  {r.status:<{status_w}}  {r.detail}")

        counts = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        print(f"\n{len(results)} model(s): {summary}")

    if args.refresh_lock:
        moved = [r for r in results if r.status == check.UPSTREAM_MOVED]
        if moved:
            _refresh(moved, models)
            print(f"Refreshed {len(moved)} lock entr(ies) whose tracked files were unchanged.")

    sys.exit(check.exit_code(results))


def _refresh(moved, models):
    """Advance the locked revision for models whose tracked files did not change."""
    from huggingface_hub import HfApi

    from exporter import check, lock

    api = HfApi()
    by_name = {m.name: m for m in models}
    entries = lock.load()
    for result in moved:
        model = by_name[result.name]
        entry = entries.get(result.name)
        if entry is None:
            continue
        info = api.model_info(model.source_repo, files_metadata=True)
        upstream = check.file_map(info)
        entry.source.revision = info.sha
        entry.tracked = {
            name: upstream[name]
            for name in model.tracked_files(list(upstream))
            if name in upstream
        }
        print(f"  {result.name}: -> {info.sha[:8]}")
    lock.save(entries)


def cmd_lock(args):
    """Reconstruct the provenance baseline for already-published models."""
    from exporter import bootstrap, lock

    if not args.bootstrap:
        entries = lock.load()
        if not entries:
            print("No lockfile yet. Run `lock --bootstrap` to create one.")
            return
        name_w = max(len(n) for n in entries)
        for name in sorted(entries):
            e = entries[name]
            rev = (e.source.revision or "—")[:8]
            print(f"{name:<{name_w}}  {e.provenance:<14}  {e.source.repo}@{rev}")
        print(f"\n{len(entries)} locked model(s) in {lock.LOCK_PATH.name}")
        return

    models = _select(args.models)
    print(f"Reconstructing provenance for {len(models)} model(s)...\n")
    entries, notes = bootstrap.bootstrap_all(models)
    for note in notes:
        print(note)

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(entries)} entr(ies) to {lock.LOCK_PATH}")
        return

    existing = lock.load()
    existing.update(entries)
    lock.save(existing)
    print(f"\nWrote {len(entries)} entr(ies) to {lock.LOCK_PATH}")


def cmd_successors(args):
    """Report upstream authors that have shipped a newer generation of a model we mirror."""
    import json as _json

    from exporter import successors

    models = _select(args.models)
    reports = successors.scan(models, top=args.top)
    with_candidates = [r for r in reports if r.candidates]

    if args.json:
        print(_json.dumps([
            {
                "model": r.name,
                "source": r.source_repo,
                "pinned_version": r.family.raw_version if r.family else None,
                "candidates": [
                    {"repo_id": c.repo_id, "version": c.version, "downloads": c.downloads,
                     "likes": c.likes, "created": c.created, "same_variant": c.variant_match}
                    for c in r.candidates
                ],
            }
            for r in (with_candidates if not args.all else reports)
        ], indent=2))
        return

    if not with_candidates:
        print("No newer generations found upstream.")
    for report in with_candidates:
        pinned = f"v{report.family.raw_version}" if report.family else "?"
        print(f"\n{report.name}  ({report.source_repo}, pinned {pinned})")
        for c in report.candidates:
            mark = "same variant" if c.variant_match else "different variant"
            created = f"created {c.created}" if c.created else ""
            print(f"    v{c.version:<5} {c.repo_id:<45} {c.downloads:>10,} dl  "
                  f"{c.likes:>4} likes  {created:<18} [{mark}]")

    checked = len(reports)
    print(f"\n{len(with_candidates)} of {checked} model(s) have a newer upstream generation.")
    print("These are name-matched candidates, not verified drop-in replacements — "
          "confirm the input/output contract before switching.")


def cmd_contract(args):
    """Show an ONNX signature, or diff a candidate against what we currently publish."""
    from pathlib import Path

    from exporter import contract

    def load(target: str) -> tuple[str, Path]:
        path = Path(target)
        if path.is_file():
            return target, path
        if path.is_dir():
            found = contract.find_onnx(path)
            if not found:
                print(f"Error: no .onnx file in {path}")
                sys.exit(2)
            return f"{target}/{found[0].name}", found[0]
        # Otherwise treat it as a model name and fetch the published file.
        model = registry.get(target)
        if model is None:
            print(f"Error: '{target}' is not a file, directory, or known model name")
            sys.exit(2)
        from huggingface_hub import hf_hub_download
        print(f"  Fetching {model.repo_id}/model.onnx ...")
        return model.repo_id, Path(hf_hub_download(model.repo_id, "model.onnx"))

    def signature(label, path):
        try:
            return contract.read(path)
        except contract.ExternalDataMissing as exc:
            print(f"Error reading {label}: {exc}")
            print("  Fetch the sidecar file next to the .onnx, then retry.")
            sys.exit(2)

    current_label, current_path = load(args.target)
    current = signature(current_label, current_path)

    if not args.against:
        print(f"{current_label}:")
        print(current.render())
        return

    cand_label, cand_path = load(args.against)
    candidate = signature(cand_label, cand_path)

    for label, path in ((current_label, current_path), (cand_label, cand_path)):
        sidecars = contract.external_data_files(path)
        if sidecars:
            print(f"note: {label} keeps weights in {', '.join(sidecars)} — "
                  f"the Java wrapper must list every one in requiredFiles\n")

    print(f"current   {current_label}:")
    print(current.render())
    print(f"\ncandidate {cand_label}:")
    print(candidate.render())

    problems = contract.diff(current, candidate)
    print()
    if not problems:
        print("Contract unchanged — safe for the existing Java wrapper.")
        return
    for p in problems:
        print(f"  {p}")
    breaking = [p for p in problems if p.startswith("BREAKING")]
    if breaking:
        print(f"\n{len(breaking)} breaking change(s): the Java wrapper needs updating "
              f"before this can be published to an existing repo id.")
        sys.exit(1)
    print("\nNo breaking changes, but review the differences above.")


def cmd_card(args):
    """Preview the model card for a model."""
    model = registry.get(args.model)
    if model is None:
        print(f"Error: unknown model '{args.model}'")
        print(f"Available models: {', '.join(registry.names())}")
        sys.exit(1)

    print(model.render_card())


def main():
    parser = argparse.ArgumentParser(
        prog="exporter",
        description="Export and mirror ONNX models for inference4j.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="Show all registered models")

    # run
    run_parser = subparsers.add_parser("run", help="Stage and upload models")
    run_parser.add_argument("models", nargs="*", help="Model names to process")
    run_parser.add_argument("--all", action="store_true", help="Process all models")
    run_parser.add_argument("--dry-run", action="store_true", help="Stage only, no upload")
    run_parser.add_argument(
        "--stage-to", metavar="DIR",
        help="Stage into an inference4j cache dir for local testing; no upload")
    run_parser.add_argument(
        "--update", action="store_true",
        help="Export from upstream HEAD instead of the locked revision, and re-pin the lock",
    )

    # check
    check_parser = subparsers.add_parser(
        "check", help="Report models whose upstream source has drifted")
    check_parser.add_argument("models", nargs="*", help="Model names (default: all)")
    check_parser.add_argument("--json", action="store_true", help="Machine-readable output")
    check_parser.add_argument(
        "--deep", action="store_true",
        help="Download files to settle comparisons metadata cannot decide")
    check_parser.add_argument(
        "--refresh-lock", action="store_true",
        help="Advance the lock for models whose repo moved but tracked files did not")

    # lock
    lock_parser = subparsers.add_parser("lock", help="Show or rebuild the provenance lockfile")
    lock_parser.add_argument("models", nargs="*", help="Model names (default: all)")
    lock_parser.add_argument(
        "--bootstrap", action="store_true",
        help="Reconstruct baselines for already-published models")
    lock_parser.add_argument("--dry-run", action="store_true", help="Preview without writing")

    # successors
    succ_parser = subparsers.add_parser(
        "successors", help="Find newer upstream generations of the models we mirror")
    succ_parser.add_argument("models", nargs="*", help="Model names (default: all)")
    succ_parser.add_argument("--json", action="store_true", help="Machine-readable output")
    succ_parser.add_argument("--top", type=int, default=3,
                             help="Candidates to show per model (default: 3)")
    succ_parser.add_argument("--all", action="store_true",
                             help="With --json, include models that have no candidates")

    # contract
    con_parser = subparsers.add_parser(
        "contract", help="Show or diff an ONNX input/output signature")
    con_parser.add_argument("target", help="Model name, staged directory, or .onnx file")
    con_parser.add_argument("--against", metavar="CANDIDATE",
                            help="Compare the target against this candidate")

    # card
    card_parser = subparsers.add_parser("card", help="Preview model card")
    card_parser.add_argument("model", help="Model name")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "run": cmd_run,
        "check": cmd_check,
        "successors": cmd_successors,
        "contract": cmd_contract,
        "lock": cmd_lock,
        "card": cmd_card,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
