"""Version-checked entry point for the native FairChem training CLI."""

import argparse
from importlib.metadata import version
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train",), default="train")
    parser.add_argument("-c", "--config", default="configs/train.yaml")
    args, overrides = parser.parse_known_args()

    if version("fairchem-core") != "2.21.0":
        raise RuntimeError("Use requirements.txt: fairchem-core must be exactly 2.21.0")
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from {root}")
    os.environ.setdefault("FAIRCHEM_CACHE_DIR", str(root.parent / "omol25-cache/fairchem"))
    os.environ.setdefault("WARP_CACHE_PATH", str(root.parent / "omol25-cache/warp"))

    import fairchem.core

    package_path = Path(fairchem.core.__file__).resolve()
    if root.parent / "fairchem" in package_path.parents:
        raise RuntimeError("Adjacent FairChem source shadows the pinned installed package")

    from fairchem.core._cli import main as fairchem_main

    fairchem_main(argparse.Namespace(config=args.config), overrides)


if __name__ == "__main__":
    main()
