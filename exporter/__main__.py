"""CLI entry point: python -m exporter list|run|card"""

import argparse
import sys

from exporter import registry
# Import models to trigger registration
import exporter.models  # noqa: F401


def cmd_list(_args):
    """Print a table of all registered models."""
    models = registry.all_models()
    if not models:
        print("No models registered.")
        return

    # Column widths
    name_w = max(len(m.name) for m in models)
    type_w = 8  # "mirrored" or "exported"
    print(f"{'Name':<{name_w}}  {'Type':<{type_w}}  Repo ID")
    print(f"{'-' * name_w}  {'-' * type_w}  {'-' * 40}")
    for m in models:
        from exporter.base import MirroredModel
        model_type = "mirrored" if isinstance(m, MirroredModel) else "exported"
        print(f"{m.name:<{name_w}}  {model_type:<{type_w}}  {m.repo_id}")


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

    print(f"Models to process: {len(models)}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    for model in models:
        upload.process(model, dry_run=args.dry_run)

    print(f"\nDone! Processed {len(models)} model(s).")


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

    # card
    card_parser = subparsers.add_parser("card", help="Preview model card")
    card_parser.add_argument("model", help="Model name")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "run": cmd_run,
        "card": cmd_card,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
