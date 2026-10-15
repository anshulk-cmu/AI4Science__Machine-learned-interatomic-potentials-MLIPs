"""Version-checked entry point for the native FairChem training CLI."""

import argparse
from importlib.metadata import distribution, version
import os
from pathlib import Path

MIN_GPU_GIB = 31   # a 32 GB V100 reports about 31.7 GiB; the settings assume at least that much


def warn_if_small_gpu():
    import torch

    if torch.cuda.is_available():
        gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if gib < MIN_GPU_GIB:
            print(f"WARNING: this GPU has {gib:.1f} GiB. Use a V100 with 32 GB (v100-32) for Dense and Global and an H100 "
                  "(h100-80) or an L40S (l40s-48) for Local-6, which peaked at 37 GiB at the default batch size.", flush=True)


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
    cache = Path(os.environ.get("OMOL_CACHE_DIR", root.parent / "omol25-cache")).resolve()
    os.environ.setdefault("FAIRCHEM_CACHE_DIR", str(cache / "fairchem"))
    os.environ.setdefault("WARP_CACHE_PATH", str(cache / "warp"))
    # less fragmentation, so fewer out-of-memory errors; OMOL_GPU_MEMORY_FRACTION caps the GPU memory for the OOM drill
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

    warn_if_small_gpu()
    if os.environ.get("OMOL_GPU_MEMORY_FRACTION"):
        import torch

        torch.cuda.set_per_process_memory_fraction(float(os.environ["OMOL_GPU_MEMORY_FRACTION"]))

    fairchem_main(argparse.Namespace(config=args.config), overrides)


if __name__ == "__main__":
    main()
