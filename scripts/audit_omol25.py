"""Audit official OMol25 ASE DB splits with the pinned FairChem reader.

Run from the fairchem-core==2.21.0 environment. No credentials are read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

EXPECTED = {"train_4M": 3_986_754, "val": 2_762_021}
SOURCES = {
    "train_4M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz",
    "val": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz",
}
SEED = 11785
SMOKE_PER_SPLIT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"Missing split directory: {path}")
    files = sorted(path.rglob("*.aselmdb"))
    if not files:
        raise ValueError(f"No .aselmdb files found in {path}")
    return files


def sample_indices(length: int, seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), min(count, length)))


def scalar_int(value: object, name: str, sample_id: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{sample_id}: {name} must be an integer, got {value!r}")
    return int(value)


def finite_array(value: object, shape: tuple[int, ...], name: str, sample_id: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{sample_id}: {name} shape/dtype {arr.shape}/{arr.dtype}, expected {shape}/number")
    if not np.isfinite(arr).all():
        raise ValueError(f"{sample_id}: {name} has NaN/Inf")
    return arr


def audit_split(name: str, split_dir: Path, seed: int, count: int, hash_file: Callable[[Path], str] | None) -> dict:
    from importlib.metadata import version

    from fairchem.core.datasets import AseDBDataset

    if version("fairchem-core") != "2.21.0":
        raise RuntimeError("Audit requires fairchem-core==2.21.0")

    files = files_under(split_dir)
    dataset = AseDBDataset(
        {
            "src": [str(path) for path in files],
            "a2g_args": {
                "molecule_cell_size": 120.0,
                "r_energy": True,
                "r_forces": True,
                "r_data_keys": ["charge", "spin"],
            },
        }
    )
    length = len(dataset)
    if length != EXPECTED[name]:
        raise ValueError(f"{name}: expected {EXPECTED[name]} rows, found {length}")
    indices = sample_indices(length, seed, count)
    samples = []
    seen_ids = set()
    atom_counts = []
    elements = set()
    charges = set()
    spins = set()
    metadata_keys = set()
    for idx in indices:
        atoms = dataset.get_atoms(idx)
        info = atoms.info
        sample_id = str(info.get("source", info.get("sid", f"{name}:{idx}")))
        if sample_id in seen_ids:
            raise ValueError(f"{name}: duplicate sampled ID {sample_id}")
        seen_ids.add(sample_id)
        n = len(atoms)
        if n < 1:
            raise ValueError(f"{sample_id}: zero atoms")
        numbers = np.asarray(atoms.numbers)
        if numbers.shape != (n,) or not np.issubdtype(numbers.dtype, np.integer):
            raise ValueError(f"{sample_id}: invalid atomic numbers")
        if np.any((numbers < 1) | (numbers > 118)):
            raise ValueError(f"{sample_id}: atomic number outside 1..118")
        positions = finite_array(atoms.positions, (n, 3), "positions", sample_id)
        if "charge" not in info or "spin" not in info:
            raise ValueError(f"{sample_id}: raw charge/spin missing; FairChem would default to 0")
        charge = scalar_int(info["charge"], "charge", sample_id)
        spin = scalar_int(info["spin"], "spin", sample_id)
        if spin < 1:
            raise ValueError(f"{sample_id}: spin multiplicity must be >=1, got {spin}")
        if "source" not in info:
            raise ValueError(f"{sample_id}: missing source ID")
        energy = float(info["energy"] if "energy" in info else atoms.get_potential_energy(apply_constraint=False))
        if not math.isfinite(energy):
            raise ValueError(f"{sample_id}: energy has NaN/Inf")
        force_values = info["forces"] if "forces" in info else atoms.get_forces(apply_constraint=False)
        forces = finite_array(force_values, (n, 3), "forces", sample_id)

        converted = dataset[idx]
        converted_numbers = converted.atomic_numbers.cpu().numpy()
        converted_positions = converted.pos.cpu().numpy()
        converted_energy = converted.energy.cpu().numpy()
        converted_forces = converted.forces.cpu().numpy()
        if not np.array_equal(converted_numbers, numbers):
            raise ValueError(f"{sample_id}: FairChem atomic numbers differ")
        # molecule_cell_size=120 centers an aperiodic molecule in a large box.
        # The absolute coordinates change, but every atom must get the same shift.
        shifts = converted_positions - positions
        if not np.allclose(shifts, shifts[0], rtol=0, atol=1e-4):
            raise ValueError(f"{sample_id}: FairChem changed relative positions")
        if not np.allclose(converted_energy, [energy], rtol=1e-6, atol=1e-4):
            raise ValueError(f"{sample_id}: FairChem energy differs")
        if not np.allclose(converted_forces, forces, rtol=1e-5, atol=1e-4):
            raise ValueError(f"{sample_id}: FairChem forces differ")
        if int(converted.charge.item()) != charge or int(converted.spin.item()) != spin:
            raise ValueError(f"{sample_id}: FairChem charge/spin differs")
        if not (np.isfinite(converted_positions).all() and np.isfinite(converted_energy).all() and np.isfinite(converted_forces).all()):
            raise ValueError(f"{sample_id}: FairChem tensors contain NaN/Inf")
        samples.append({"index": idx, "source_id": sample_id})
        atom_counts.append(n)
        elements.update(int(z) for z in numbers)
        charges.add(charge)
        spins.add(spin)
        metadata_keys.update(info.keys())
    return {
        "split": name,
        "archive_url": SOURCES[name],
        "path": str(split_dir),
        "row_count": length,
        "files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hash_file(path) if hash_file else None}
            for path in files
        ],
        "sample_indices": indices,
        "samples": samples,
        "sample_checks": {
            "count": len(samples),
            "atomic_numbers": "integer 1..118; raw and FairChem agree",
            "positions": "finite, Nx3; FairChem differs only by uniform box-centering translation",
            "energy": "finite scalar; raw and FairChem agree",
            "forces": "finite Nx3; raw and FairChem agree",
            "charge_spin": "present in raw atoms.info; integer; FairChem agrees",
            "source_id": "present and unique within sampled split",
            "natoms_min": min(atom_counts),
            "natoms_max": max(atom_counts),
            "elements_observed": sorted(elements),
            "charges_observed": sorted(charges),
            "spin_multiplicities_observed": sorted(spins),
            "metadata_keys_observed": sorted(metadata_keys),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke-per-split", type=int, default=SMOKE_PER_SPLIT)
    parser.add_argument("--skip-hashes", action="store_true", help="Fast sample precheck only; output status remains pending")
    parser.add_argument("--source-revision", required=True, help="Hugging Face commit SHA of the official DATASET.md download instructions")
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    fairchem = repo.parent / "fairchem"
    if data_root == repo or repo in data_root.parents or data_root == fairchem or fairchem in data_root.parents:
        raise ValueError("Data must live outside mlip-project and fairchem Git repositories")
    if args.smoke_per_split < 1:
        raise ValueError("--smoke-per-split must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision):
        raise ValueError("--source-revision must be a full 40-character Hugging Face commit SHA")

    # Hashing all 160 database shards plus archives can take a long time on a
    # local disk. Save each finished hash outside Git so a restart can resume.
    cache_path = data_root / "sha256-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}

    def cached_hash(path: Path) -> str:
        stat = path.stat()
        key = str(path)
        entry = cache.get(key)
        if entry and entry["size_bytes"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
            return entry["sha256"]
        digest = sha256(path)
        cache[key] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        temp_cache = cache_path.with_suffix(".json.tmp")
        temp_cache.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_cache, cache_path)
        print(f"hashed {path.name}: {digest}", flush=True)
        return digest

    hash_file = None if args.skip_hashes else cached_hash
    report = {
        "status": "samples_verified_hashes_pending" if args.skip_hashes else "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": "https://huggingface.co/facebook/OMol25",
        "source_version": "OMol-0",
        "source_revision": args.source_revision,
        "fairchem_core": "2.21.0",
        "fairchem_checkout_commit": "db9b17fc6eff8bd167fc2174fddc8f43bab34629",
        "seed": args.seed,
        "smoke_per_split": args.smoke_per_split,
        "units": {"positions": "angstrom", "energy": "eV", "forces": "eV/angstrom"},
        "splits": [
            audit_split("train_4M", data_root / "train_4M", args.seed, args.smoke_per_split, hash_file),
            audit_split("val", data_root / "val", args.seed + 1, args.smoke_per_split, hash_file),
        ],
        "archives": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hash_file(path) if hash_file else None}
            for path in (data_root / "train_4M.tar.gz", data_root / "val.tar.gz")
            if path.is_file()
        ],
    }
    train_ids = {row["source_id"] for row in report["splits"][0]["samples"]}
    val_ids = {row["source_id"] for row in report["splits"][1]["samples"]}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Sampled source IDs overlap across official splits: {sorted(overlap)[:5]}")
    report["sampled_cross_split_id_overlap"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Verified {len(report['splits'])} splits; wrote {args.output}")


if __name__ == "__main__":
    main()
