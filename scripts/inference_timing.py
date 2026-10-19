"""Inference timing of the final pilot checkpoints (Dense, Global, Local-6) on one GPU.

For each checkpoint and each fixed input, three things are timed, each with warm-up calls first and
torch.cuda.synchronize() around every timed call:
  graph           backbone._generate_graph alone (the radius graph built on the GPU, 6 A, 300 neighbours)
  energy          one model forward under no_grad, with the forces head switched off
  energy_forces   one model forward with the forces from autograd, as in validation (direct_forces is false)
The batch is on the GPU before the clock starts; the host-to-device copy and the denormalisation are not timed.
Inputs: one packed batch per size bin (the structures of the validation list whose atom count is closest to
the target, as many as fit in max_atoms) and the first validation batches in row order (the validation batching).

  run:   python scripts/inference_timing.py run --out results/inference_timing_v100.csv
  plot:  python scripts/inference_timing.py plot results/inference_timing_v100.csv [more CSVs]
"""
import argparse
import csv
import gc
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = "/ocean/projects/che260063p/shared"
VARIANTS = ("dense", "global", "local6")
MODES = ("graph", "energy", "energy_forces")
NAMES = {"dense": "Dense", "global": "Global", "local6": "Local-6"}


# ---------- pure helpers (tested in tests/test_inference_timing.py) ----------

def pick_bin(rows, natoms, target, max_atoms):
    """Rows of the structures closest in size to target (ties: lower row), as many as fit in max_atoms."""
    order = sorted(range(len(rows)), key=lambda i: (abs(natoms[i] - target), rows[i]))
    picked, total = [], 0
    for i in order[: max(1, max_atoms // target)]:
        if picked and total + natoms[i] > max_atoms:
            break
        picked.append(i)
        total += natoms[i]
    return sorted(int(rows[i]) for i in picked)


def pack(rows, natoms, max_atoms):
    """Consecutive groups of rows with at most max_atoms atoms each, as the validation batches."""
    groups, group, n = [], [], 0
    for r, k in zip(rows, natoms):
        if group and n + k > max_atoms:
            groups.append(group)
            group, n = [], 0
        group.append(int(r))
        n += k
    return groups + [group] if group else groups


def quartiles(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(q1), float(q3)


def timed(prepare, call, warmup, repeats, sync):
    """Seconds per call; prepare() runs outside the clock, warm-up calls are not returned."""
    for _ in range(warmup):
        call(prepare())
    out = []
    for _ in range(repeats):
        x = prepare()
        sync()
        t = time.perf_counter()
        call(x)
        sync()
        out.append(time.perf_counter() - t)
    return np.array(out)


def param_counts(model):
    """Total parameters and the active count per structure: one of the K expert slices of every MOLE layer."""
    from fairchem.core.models.uma.nn.mole import MOLE
    total = sum(p.numel() for p in model.parameters())
    active = total
    for m in model.modules():
        if isinstance(m, MOLE):
            k = m.weights.shape[0]
            active -= m.weights.numel() * (k - 1) // k
    return total, active


@contextmanager
def forces_off(backbone):
    """Energy only: the backbone and the head share one regress_config, so this drops the autograd forces."""
    cfg = backbone.regress_config
    old = cfg.forces
    cfg.forces = False
    try:
        yield
    finally:
        cfg.forces = old


# ---------- model and data ----------

def load_unit(path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))            # the Local-6 checkpoint names classes in training/ and local_routing/
    import torch
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    # the router-check settings: float32, Euler-angle Wigner as in the checkpoint, general backend, no stress task
    settings = InferenceSettings(base_precision_dtype=torch.float32, use_quaternion_wigner=None, execution_mode="general",
                                 auto_add_default_untrained_tasks=False)
    return pretrained_mlip.load_predict_unit(str(path), inference_settings=settings, device="cuda"), settings


def to_batch(db, rows):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    items = [AtomicData.from_ase(db.get_atoms(r), task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                                 molecule_cell_size=120.0) for r in rows]
    return atomicdata_list_to_batch(items)


def inputs(src, rows_file, targets, val_batches, max_atoms):
    """{condition: [batch, ...]} and the number of candidate rows."""
    from fairchem.core.datasets import AseDBDataset
    db = AseDBDataset({"src": str(src)})
    rows = np.loadtxt(rows_file, dtype=np.int64, ndmin=1)
    rows = rows[rows < len(db)]                  # all 5,000 on Bridges-2; the laptop copy holds one shard only
    meta = Path(src) / "metadata.npz"
    natoms = np.load(meta)["natoms"][rows] if meta.exists() else np.array([len(db.get_atoms(int(r))) for r in rows])
    conds = {f"bin{t}": [to_batch(db, pick_bin(rows, natoms, t, max_atoms))] for t in targets}
    conds["val"] = [to_batch(db, g) for g in pack(rows, natoms, max_atoms)[:val_batches]]
    return conds, len(rows)


def checkpoints(a):
    """[(variant, seed, path)] from --ckpt VARIANT:SEED:PATH, or the final checkpoints of the pilot runs."""
    if a.ckpt:
        return [(v, int(s), Path(p)) for v, s, p in (c.split(":", 2) for c in a.ckpt)]
    out = []
    for s in a.seeds:                            # seed 0 of all variants first, in case the time limit hits
        for v in a.variants:
            runs = [p for p in Path(a.runs_root).glob(f"pilot_{v}_s{s}_*") if re.fullmatch(rf"pilot_{v}_s{s}_\d+", p.name)]
            assert len(runs) == 1, f"expected one run folder for {v} seed {s}, got {runs}"
            ck = list(runs[0].glob("train/*/checkpoints/final/inference_ckpt.pt"))
            assert len(ck) == 1, ck
            out.append((v, s, ck[0]))
    return out


def environment():
    import torch
    import fairchem.core
    from importlib.metadata import version
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, encoding="utf-8", timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        smi = []
    return dict(gpu=torch.cuda.get_device_name(0), driver=smi[0].split(", ")[1] if smi else "",
                gpu_count_visible=torch.cuda.device_count(), host=platform.node(), torch=torch.__version__,
                cuda=torch.version.cuda, fairchem=version("fairchem-core"), python=platform.python_version(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(), slurm_job=os.environ.get("SLURM_JOB_ID", ""))


# ---------- run ----------

def measure(unit, settings, conds, warmup, repeats):
    import torch
    model, backbone = unit.model, unit.model.module.backbone
    dtype = settings.base_precision_dtype
    sync = torch.cuda.synchronize
    unit.predict(conds["val"][0], undo_element_references=False)          # lazy init: dtype, device, inference prep

    def on_gpu(batch):
        d = batch.to(unit.device)
        for key, val in d:
            if torch.is_tensor(val) and val.is_floating_point():
                d[key] = val.to(dtype)
        return d

    def graph(xs):
        with torch.no_grad():
            for x in graph_inputs(xs):
                backbone._generate_graph(x)

    def energy(xs):
        with forces_off(backbone), torch.no_grad():
            return [model(x)["omol_energy"]["energy"] for x in xs]

    def energy_forces(xs):
        return [(o["omol_energy"]["energy"].detach(), o["omol_forces"]["forces"].detach()) for o in map(model, xs)]

    calls = dict(graph=graph, energy=energy, energy_forces=energy_forces)
    rows = []
    for cond, batches in conds.items():
        dev = [on_gpu(b) for b in batches]

        def prepare():
            return [d.clone() for d in dev]

        with torch.no_grad():
            edges = sum(backbone._generate_graph(x)["edge_index"].shape[1] for x in graph_inputs(prepare()))
        e_only = torch.cat(energy(prepare()))
        e_f = energy_forces(prepare())
        # self-check: the energy-only path computes the same energy, and the forces exist and are finite
        assert torch.allclose(e_only, torch.cat([e for e, _ in e_f]), rtol=1e-4, atol=1e-4), (e_only, e_f)
        assert all(torch.isfinite(f).all() and f.shape[1] == 3 for _, f in e_f)
        n_atoms = sum(int(b.natoms.sum()) for b in batches)
        n_struct = sum(len(b.natoms) for b in batches)
        for mode in MODES:
            sync()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t = timed(prepare, calls[mode], warmup, repeats, sync) / len(batches)       # seconds per batch
            peak = torch.cuda.max_memory_allocated()
            med, q1, q3 = quartiles(t)
            aps = quartiles(n_atoms / len(batches) / t)
            rows.append(dict(condition=cond, mode=mode, batches=len(batches), structures=n_struct, atoms=n_atoms,
                             atoms_per_batch=n_atoms / len(batches), edges=edges, repeats=repeats, warmup=warmup,
                             sec_per_batch_median=med, sec_per_batch_q1=q1, sec_per_batch_q3=q3,
                             atoms_per_s_median=aps[0], atoms_per_s_q1=aps[1], atoms_per_s_q3=aps[2],
                             peak_mem_gib=peak / 2**30, base_mem_gib=base / 2**30))
            print(f"  {cond:7s} {mode:13s} {n_atoms:5d} atoms {edges:7d} edges: {med * 1e3:8.2f} ms per batch "
                  f"(IQR {q1 * 1e3:.2f} to {q3 * 1e3:.2f}), {aps[0]:9.0f} atoms/s, peak {peak / 2**30:.2f} GiB", flush=True)
    return rows


def graph_inputs(xs):
    """The keys that eSCNMDBackbone.forward sets before it builds the graph."""
    for x in xs:
        x["atomic_numbers"] = x["atomic_numbers"].long()
        x["atomic_numbers_full"], x["batch_full"] = x["atomic_numbers"], x["batch"]
    return xs


def run(a):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    env = environment()
    print(env, flush=True)
    t0 = time.time()
    conds, n_rows = inputs(a.src, a.rows, a.targets, a.val_batches, a.max_atoms)
    print(f"{n_rows} candidate rows; conditions " + ", ".join(
        f"{c}: {sum(len(b.natoms) for b in bs)} structures, {sum(int(b.natoms.sum()) for b in bs)} atoms"
        for c, bs in conds.items()) + f"; loaded in {time.time() - t0:.0f} s", flush=True)
    out = []
    for variant, seed, path in checkpoints(a):
        t = time.time()
        unit, settings = load_unit(path)
        total, active = param_counts(unit.model.module)
        bb = unit.model.module.backbone
        tag = dict(variant=variant, seed=seed, params_total=total, params_active=active, cutoff=bb.cutoff,
                   max_neighbors=bb.max_neighbors, otf_graph=bb.otf_graph,
                   edge_chunk=next((m.activation_checkpoint_chunk_size for m in bb.modules()
                                    if getattr(m, "activation_checkpoint_chunk_size", None)), None),
                   dtype=str(settings.base_precision_dtype).replace("torch.", ""), execution_mode=settings.execution_mode,
                   compile=settings.compile, candidate_rows=n_rows, checkpoint=str(path))
        print(f"{variant} s{seed}: {path} ({total:,} parameters, {active:,} active)", flush=True)
        out += [{**tag, **r, **env} for r in measure(unit, settings, conds, a.warmup, a.repeats)]
        print(f"{variant} s{seed} done in {time.time() - t:.0f} s", flush=True)
        del unit, bb
        gc.collect()
        torch.cuda.empty_cache()
        write(a.out, out)                        # after every checkpoint, so a time-out keeps what finished


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


# ---------- plot and tables ----------

def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    frames = [pd.read_csv(p) for p in a.csv]
    colors = {"dense": "#4C72B0", "global": "#C44E52", "local6": "#55A868"}
    fig, axes = plt.subplots(len(frames), 2, figsize=(13, 4.3 * len(frames)), squeeze=False)
    for (ax1, ax2), df in zip(axes, frames):
        df = df[df.seed == df.seed.min()]
        conds = list(dict.fromkeys(df.condition))
        labels = [f"{c}\n{int(df[df.condition == c].atoms_per_batch.iloc[0])} at." for c in conds]
        x = np.arange(len(conds))
        for j, (v, c) in enumerate(colors.items()):
            for mode, ls, mk in (("energy_forces", "-", "o"), ("energy", "--", "s")):
                d = df[(df.variant == v) & (df["mode"] == mode)].set_index("condition").reindex(conds)
                xs = x + (j - 1) * 0.08
                ax1.errorbar(xs, d.atoms_per_s_median, yerr=[d.atoms_per_s_median - d.atoms_per_s_q1,
                             d.atoms_per_s_q3 - d.atoms_per_s_median], color=c, ls=ls, marker=mk, ms=4, capsize=2,
                             label=f"{NAMES[v]}, {mode.replace('_', ' + ')}")
                ax2.plot(xs, d.peak_mem_gib, color=c, ls=ls, marker=mk, ms=4, label=f"{NAMES[v]}, {mode.replace('_', ' + ')}")
        gpu = df.gpu.iloc[0]
        ax1.set_yscale("log")
        ax1.set_ylabel("atoms per second (median, IQR bars)")
        ax2.set_ylabel("peak GPU memory allocated (GiB)")
        for ax in (ax1, ax2):
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(f"{gpu}, seed 0, float32, batch on the GPU", fontsize=9)
            ax.grid(alpha=0.3)
        ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "inference_timing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for df in frames:
        tables(df)


def tables(df):
    """Markdown tables for docs/inference-timing.md: seed 0, Local-6 / Global ratios, seed spread."""
    print(f"\n### {df.gpu.iloc[0]} (driver {df.driver.iloc[0]}, host {df.host.iloc[0]}, job {df.slurm_job.iloc[0]})\n")
    s0 = df[df.seed == df.seed.min()]
    print("| Condition | Atoms | Edges | Mode | " + " | ".join(f"{NAMES[v]} ms (IQR)" for v in VARIANTS)
          + " | " + " | ".join(f"{NAMES[v]} atoms/s" for v in VARIANTS) + " | " + " | ".join(f"{NAMES[v]} GiB" for v in VARIANTS) + " |")
    print("|" + "---|" * 13)
    for (cond, mode), g in s0.groupby(["condition", "mode"], sort=False):
        g = g.set_index("variant")
        ms = [f"{g.loc[v].sec_per_batch_median * 1e3:.1f} ({g.loc[v].sec_per_batch_q1 * 1e3:.1f} to {g.loc[v].sec_per_batch_q3 * 1e3:.1f})" for v in VARIANTS]
        aps = [f"{g.loc[v].atoms_per_s_median:,.0f}" for v in VARIANTS]
        mem = [f"{g.loc[v].peak_mem_gib:.2f}" for v in VARIANTS]
        r = g.iloc[0]
        print(f"| {cond} | {r.atoms_per_batch:.0f} | {r.edges / r.batches:.0f} | {mode} | " + " | ".join(ms + aps + mem) + " |")
    print("\nRatios of the median time per batch, seed 0:\n")
    print("| Condition | Mode | Local-6 / Global | Global / Dense | Local-6 / Dense | Local-6 / Global peak memory |")
    print("|---|---|---:|---:|---:|---:|")
    for (cond, mode), g in s0.groupby(["condition", "mode"], sort=False):
        t, m = g.set_index("variant").sec_per_batch_median, g.set_index("variant").peak_mem_gib
        print(f"| {cond} | {mode} | {t.local6 / t['global']:.2f} | {t['global'] / t.dense:.2f} | {t.local6 / t.dense:.2f} | "
              f"{m.local6 / m['global']:.2f} |")
    if df.seed.nunique() > 1:
        print("\nSeed spread of the median time per batch (ms), min to max over seeds:\n")
        print("| Condition | Mode | " + " | ".join(NAMES[v] for v in VARIANTS) + " |")
        print("|---|---|---|---|---|")
        for (cond, mode), g in df.groupby(["condition", "mode"], sort=False):
            cells = []
            for v in VARIANTS:
                s = g[g.variant == v].sec_per_batch_median * 1e3
                cells.append(f"{s.min():.1f} to {s.max():.1f} (n={len(s)})")
            print(f"| {cond} | {mode} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", action="append", help="VARIANT:SEED:PATH (repeatable); default: the pilot final checkpoints")
    p.add_argument("--runs-root", default=f"{SHARED}/runs")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--src", default=f"{SHARED}/data/omol25/val")
    p.add_argument("--rows", default=str(ROOT / "manifests" / "pilot_val_5k.txt"))
    p.add_argument("--targets", type=int, nargs="+", default=[25, 50, 100, 200, 340], help="size bins, atoms per structure")
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--max-atoms", type=int, default=450)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=25)
    p.set_defaults(fn=run)
    p = sub.add_parser("plot")
    p.add_argument("csv", nargs="+")
    p.set_defaults(fn=plot)
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.repeats >= 20 and a.warmup >= 1, "at least 20 timed repeats and one warm-up call"
    a.fn(a)


if __name__ == "__main__":
    main()
