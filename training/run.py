"""Version-checked entry point for the native FairChem training CLI."""

import argparse
from importlib.metadata import distribution, version
import os
from pathlib import Path

BASE_ATOMS = 320   # training.max_atoms in configs/train.yaml
# Atoms per batch, as a multiple of the base, chosen to fill a 32 GB GPU: the smoke test peaked at 7.1 GiB (Dense),
# 7.9 GiB (Global) and 19.3 GiB (Local-6) with 320 atoms. Pass training.max_atoms=N to override.
BATCH_SCALE = {"dense": 2.5, "global": 2.5, "local6": 1.25}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train",), default="train")
    parser.add_argument("-c", "--config", default="configs/train.yaml")
    args, overrides = parser.parse_known_args()

    variant = next((o.split("=", 1)[1] for o in overrides if o.startswith("variant=")), "dense")
    if variant in BATCH_SCALE and not any(o.startswith("training.max_atoms=") for o in overrides):
        overrides.append(f"training.max_atoms={round(BASE_ATOMS * BATCH_SCALE[variant])}")

    if version("fairchem-core") != "2.21.0":
        raise RuntimeError("Use requirements.txt: fairchem-core must be exactly 2.21.0")
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from {root}")
    cache = Path(os.environ.get("OMOL_CACHE_DIR", root.parent / "omol25-cache")).resolve()
    os.environ.setdefault("FAIRCHEM_CACHE_DIR", str(cache / "fairchem"))
    os.environ.setdefault("WARP_CACHE_PATH", str(cache / "warp"))
    # W&B stays local: runs are written under the run directory on Ocean and uploaded later with `wandb sync`
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache / "wandb"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(cache / "wandb_config"))

    import fairchem.core

    package_path = Path(fairchem.core.__file__).resolve()
    installed = Path(distribution("fairchem-core").locate_file("fairchem/core/__init__.py")).resolve()
    if package_path != installed:
        raise RuntimeError(f"fairchem.core imports from {package_path}, not the installed {installed}")

    from fairchem.core._cli import main as fairchem_main

    fairchem_main(argparse.Namespace(config=args.config), overrides)


if __name__ == "__main__":
    main()
