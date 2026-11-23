"""Pinned, isolated environment for the RTMO (MMPose) exports.

MMPose 1.3.2 and MMDeploy 1.3.1 need torch 2.1 and mmcv 2.1, which in turn need Python 3.10, while the
project's main .venv runs a current Python and torch. The export therefore runs in its own venv,
`.venv-rtmo`, created on first use with `uv` (which also provides the Python 3.10 interpreter):

1. build prerequisites (numpy, torch, cython, ninja) from `rtmo_requirements.txt`;
2. mmcv 2.1.0 compiled from its sdist with the CPU ops (no macOS or recent Linux wheel exists; MMDeploy's
   mmdet rewriters import `mmcv.ops`);
3. everything else in `rtmo_requirements.txt`, with `--no-deps` so nothing is re-resolved.

MMDeploy 1.3.1 has no wheel for macOS and its setup.py compiles extensions we do not need, so the export
script puts its source tree (a pinned GitHub tarball, verified by sha256) on `sys.path` instead.
"""

import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTMO_VENV = ROOT / ".venv-rtmo"
REQUIREMENTS = Path(__file__).with_name("rtmo_requirements.txt")
EXPORT_SCRIPT = Path(__file__).with_name("rtmo_export_script.py")
PYTHON_VERSION = "3.10"

MMCV_SDIST = "https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-2.1.0.tar.gz"
# torch 2.1 headers specialise std::is_arithmetic, which recent clang rejects as an error.
MMCV_BUILD_ENV = {
    "MMCV_WITH_OPS": "1",
    "CFLAGS": "-Wno-invalid-specialization",
    "CXXFLAGS": "-Wno-invalid-specialization",
}
_BUILD_PREREQS = ("setuptools", "wheel", "cython", "numpy", "torch", "ninja")

MMDEPLOY_TARBALL = "https://github.com/open-mmlab/mmdeploy/archive/refs/tags/v1.3.1.tar.gz"
MMDEPLOY_SHA256 = "20c6111c65e336d40605a243df02b94440217deaaaa41c30814580773bf99a59"
CACHE_DIR = Path.home() / ".cache" / "inference4j" / "rtmo"


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("The RTMO export needs `uv` (https://docs.astral.sh/uv/) to create .venv-rtmo")
    return uv


def _pins() -> list[str]:
    return [
        line.strip() for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def get_python() -> Path:
    """Return .venv-rtmo's interpreter, creating or rebuilding the venv when the pins change."""
    python = RTMO_VENV / "bin" / "python"
    stamp = RTMO_VENV / "inference4j-requirements.txt"
    wanted = REQUIREMENTS.read_text() + f"mmcv @ {MMCV_SDIST}\n"
    if python.exists() and stamp.exists() and stamp.read_text() == wanted:
        return python

    uv = _uv()
    print(f"  Creating RTMO export venv at {RTMO_VENV} (Python {PYTHON_VERSION}) ...")
    subprocess.run([uv, "venv", "--clear", "--python", PYTHON_VERSION, str(RTMO_VENV)], check=True)
    pins = _pins()
    prereqs = [p for p in pins if p.split("==")[0].lower() in _BUILD_PREREQS]
    install = [uv, "pip", "install", "--python", str(python), "--no-deps"]
    subprocess.run([*install, *prereqs], check=True)
    subprocess.run([*install, "--no-build-isolation", f"mmcv @ {MMCV_SDIST}"], check=True,
                   env={**os.environ, **MMCV_BUILD_ENV})
    subprocess.run([*install, "--no-build-isolation", *pins], check=True)
    stamp.write_text(wanted)
    return python


def download(url: str, sha256: str, dest: Path) -> Path:
    """Download `url` to `dest` once; refuse bytes whose sha256 differs from the pin."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading {url} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    digest = hashlib.sha256()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"{dest.name}: sha256 {digest.hexdigest()} does not match the pinned {sha256}")
    return dest


def mmdeploy_root() -> Path:
    """The pinned MMDeploy 1.3.1 source tree, downloaded and extracted once."""
    root = CACHE_DIR / "mmdeploy-1.3.1"
    if not (root / "mmdeploy").is_dir():
        tarball = download(MMDEPLOY_TARBALL, MMDEPLOY_SHA256, CACHE_DIR / "mmdeploy-1.3.1.tar.gz")
        with tarfile.open(tarball) as tar:
            tar.extractall(CACHE_DIR, filter="data")
    return root


def export(config: str, checkpoint: Path, output: Path) -> None:
    """Run the MMDeploy export of one checkpoint inside .venv-rtmo."""
    python = get_python()
    subprocess.run(
        [str(python), "-I", str(EXPORT_SCRIPT),
         "--mmdeploy-root", str(mmdeploy_root()),
         "--config", config,
         "--checkpoint", str(checkpoint),
         "--output", str(output)],
        check=True,
    )
