"""Molecular locality and finite-difference checks for trained OMol checkpoints."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from importlib.metadata import version

import numpy as np
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("H2O", "CH4"), ("NH3", "CO2"), ("H2O", "H2O"))


def fragments(name_a, name_b, gap):
    """Nonperiodic neutral singlets; gap is the separation of x bounding planes."""
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("gap must be finite and positive")
    a, b = molecule(name_a), molecule(name_b)
    for atoms in (a, b):
        atoms.rotate(23, (0.3, 0.8, 0.5))
        atoms.rotate(41, (1, -0.2, 0.6))
        atoms.info.update(charge=0, spin=1)
        atoms.pbc = False
    b.translate([a.positions[:, 0].max() - b.positions[:, 0].min() + gap, 0, 0])
    pair = a + b
    pair.info.update(charge=0, spin=1)
    # Same cell and exact coordinates in isolated and combined evaluations.
    span = np.ptp(pair.positions, axis=0)
    for atoms in (a, b, pair):
        atoms.set_cell(np.diag(span + 40))
    return a, b, pair


def predict(energy, atoms):
    e = float(energy(atoms))
    f = np.asarray(energy.forces(atoms), dtype=float)
    if not math.isfinite(e) or f.shape != (len(atoms), 3) or not np.isfinite(f).all():
        raise ValueError("predictor returned invalid energy or forces")
    return e, f


def locality(energy, a, b, pair, cutoff):
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if any(atoms.pbc.any() for atoms in (a, b, pair)):
        raise ValueError("molecular evaluation requires nonperiodic inputs")
    if not (np.array_equal(pair.numbers, np.r_[a.numbers, b.numbers]) and
            np.array_equal(pair.positions, np.concatenate((a.positions, b.positions)))):
        raise ValueError("pair must preserve the isolated fragments and atom order")
    minimum = np.linalg.norm(a.positions[:, None] - b.positions[None, :], axis=-1).min()
    if minimum <= cutoff:
        raise ValueError("fragments are connected within the cutoff")
    radius = max(cutoff, getattr(energy, "routing_radius", 0.0))
    ea, fa = predict(energy, a)
    eb, fb = predict(energy, b)
    ep, fp = predict(energy, pair)
    ea2, fa2 = predict(energy, a)
    eb2, fb2 = predict(energy, b)
    ep2, fp2 = predict(energy, pair)
    change = np.linalg.norm(fp - np.concatenate((fa, fb)), axis=1)
    return dict(interaction_radius_A=radius, routing_disconnected=bool(minimum > radius + 1.0),
                min_distance_A=float(minimum), e_a_eV=ea, e_b_eV=eb, e_pair_eV=ep,
                delta_eV=ep-ea-eb,
                max_dforce_a_eV_A=float(change[:len(a)].max()),
                max_dforce_b_eV_A=float(change[len(a):].max()),
                repeat_energy_eV=max(abs(ea2-ea), abs(eb2-eb), abs(ep2-ep)),
                repeat_force_eV_A=max(float(np.linalg.norm(y-x, axis=1).max())
                                     for x, y in ((fa, fa2), (fb, fb2), (fp, fp2))))


def finite_difference(energy, atoms, step):
    """Central differences for every Cartesian coordinate; no force-derived energy."""
    if not math.isfinite(step) or step <= 0:
        raise ValueError("finite-difference step must be finite and positive")
    _, forces = predict(energy, atoms)
    rows = []
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            ep, em = float(energy(plus)), float(energy(minus))
            if not np.isfinite([ep, em]).all():
                raise ValueError("nonfinite finite-difference energy")
            numeric = -(ep-em)/(2*step)
            rows.append(dict(atom=atom, axis=axis, step_A=step,
                             force_eV_A=float(forces[atom, axis]), fd_force_eV_A=numeric,
                             abs_error_eV_A=abs(numeric-forces[atom, axis])))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    """Output of a git command in the repository, or None: the Bridges-2 copy made by syncrepo has no .git."""
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path, help="local fairchem inference checkpoint")
    ap.add_argument("--run-config", type=Path, required=True, help="resolved config from this training run")
    ap.add_argument("--label", required=True, help="e.g. local6_seed0_step50000")
    ap.add_argument("--output", type=Path, required=True, help="new output directory; never overwritten")
    ap.add_argument("--gaps", type=float, nargs="+", default=[8, 10, 15, 25, 100])
    ap.add_argument("--fd-steps", type=float, nargs="+", default=[0.001, 0.0003])
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = ap.parse_args()
    for path in (args.checkpoint, args.run_config):
        if not path.is_file():
            ap.error(f"file does not exist: {path}")
    if any(not math.isfinite(x) or x <= 0 for x in args.gaps + args.fd_steps):
        ap.error("gaps and finite-difference steps must be finite and positive")
    if version("fairchem-core") != "2.21.0":
        ap.error("use the team's fairchem-core==2.21.0 environment")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(label=args.label, status="started", task="omol", charge=0, spin=1,
                    dtype=args.dtype, gaps_A=args.gaps, fd_steps_A=args.fd_steps,
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                    run_config=str(args.run_config.resolve()), run_config_sha256=sha256(args.run_config),
                    resolved_config_text=args.run_config.read_text(encoding="utf-8"),
                    evaluation_commit=(git("rev-parse", "HEAD") or "").strip() or None,
                    working_tree_status=git("status", "--porcelain"),
                    versions={p: version(p) for p in ("numpy", "ase", "torch", "fairchem-core")})
    meta_path = args.output / "metadata.json"
    def save():
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        # Existing loader resets the ASE cache and uses standard, unmerged inference.
        from audit.models import load
        energy, cutoff = load("uma", str(args.checkpoint.resolve()), "omol", args.dtype)
        metadata["cutoff_A"] = cutoff
        metadata["routing_radius_A"] = getattr(energy, "routing_radius", 0.0)
        # unit.model is an AveragedModel wrapper; the backbone class names the variant
        metadata["backbone_class"] = next((type(m.backbone).__module__ + "." + type(m.backbone).__name__
                                           for m in energy.unit.model.modules() if hasattr(m, "backbone")), None)
        metadata["routing_modules"] = sorted({type(m).__module__ + "." + type(m).__name__
                                              for m in energy.unit.model.modules()
                                              if hasattr(m, "set_MOLE_coefficients")})
        rows, fd = [], []
        geometries = []
        for name_a, name_b in PAIRS:
            for gap in args.gaps:
                a, b, pair = fragments(name_a, name_b, gap)
                name = f"{name_a}+{name_b}"
                rows.append(dict(label=args.label, pair=name, gap_A=gap,
                                 **locality(energy, a, b, pair, cutoff)))
                geometries.append(dict(pair=name, gap_A=gap, numbers=pair.numbers.tolist(),
                                       positions_A=pair.positions.tolist(), cell_A=pair.cell.array.tolist()))
        # Each isolated molecule once: geometry-dependent routing acts inside bonded molecules.
        for name in dict.fromkeys(n for p in PAIRS for n in p):
            atoms = fragments(name, name, args.gaps[0])[0]
            for step in args.fd_steps:
                fd.extend(dict(label=args.label, molecule=name, **r)
                          for r in finite_difference(energy, atoms, step))
        write_csv(args.output / "locality.csv", rows)
        write_csv(args.output / "finite_difference.csv", fd)
        (args.output / "geometries.json").write_text(json.dumps(geometries, indent=2) + "\n", encoding="utf-8")
        metadata["status"] = "complete"
        save()
        print(f"Wrote {len(rows)} locality rows and {len(fd)} force checks to {args.output}")
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
