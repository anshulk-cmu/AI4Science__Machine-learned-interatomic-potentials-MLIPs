"""Router check: does the MoLE router of the Global and Local-6 pilots do any work?

Three measurements on the pilot checkpoints, on the fixed validation list:
  router    statistics of the router output p = softmax(logits) per routed unit (a structure for Global, an atom for
            Local-6): entropy, max, L1 and KL to uniform, spread across units and, for Local-6, inside each structure.
            One router feeds all 24 MoLE layers (one MOLEGlobals object), so these hold for every layer.
  experts   per MoLE layer: pairwise cosine of the 8 expert weight slices, their spread around the expert mean, and,
            against step 0 of the same run, the same numbers for the training updates W_k(step) - W_k(0).
  ablation  energy MAE per atom and force MAE with the router output replaced by a fixed vector (the same for every
            structure and atom): 'uniform' (equal logits, alpha = 1/8 + 0.005 = 0.13 per expert, the router's own
            uniform point; fairchem's softmax norm adds 0.005, so alpha always sums to 1.04), 'eighth' (alpha = 1/8,
            sum 1.00) and 'mean' (the mean router output over the validation units), against 'trained'.

  run:      python scripts/router_check.py run --run-dir RUNS/pilot_global_s0_49041939 --out-dir DIR
  analyze:  python scripts/router_check.py analyze --in-dir DIR      (merges the per-run CSVs into results/, tests, figure)
"""
import argparse
import csv
import math
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 8
CONDITIONS = ("trained", "uniform", "eighth", "mean")


# ---------- router ----------

def route(backbone, data):
    """Router output alpha [units, K] for a batch, without the message passing; mirrors eSCNMDBackbone.forward."""
    data["atomic_numbers"] = data["atomic_numbers"].long()
    data["atomic_numbers_full"] = data["atomic_numbers"]
    data["batch_full"] = data["batch"]
    csd = backbone.csd_embedding(charge=data["charge"], spin=data["spin"], dataset=data.get("dataset", default=None))
    backbone.set_MOLE_coefficients(data["atomic_numbers_full"], data["batch_full"], csd)
    if hasattr(backbone, "routing_radius"):          # Local: routes once the graph exists
        graph = backbone._generate_graph(data)
        backbone.set_MOLE_sizes(len(data["natoms"]), data["batch_full"], graph["edge_index"])
    return backbone.global_mole_tensors.expert_mixing_coefficients.detach()


@contextmanager
def fixed_alpha(backbone, alpha=None, record=None):
    """Replace the router output by alpha [K] for every unit (None keeps the trained router); record what is used."""
    orig = backbone.mole_expert_coefficient_norm
    import torch

    def norm(x):
        out = orig(x) if alpha is None else torch.as_tensor(alpha).to(x).expand(x.shape[0], -1)
        if record is not None:
            record.append(out.detach())
        return out

    backbone.mole_expert_coefficient_norm = norm
    try:
        yield
    finally:
        backbone.mole_expert_coefficient_norm = orig


def softmax_part(alpha):
    """fairchem's norm is softmax(z) + 0.005; return the softmax, a distribution over the K experts."""
    return alpha - 0.005


def unit_stats(p):
    """Per unit: entropy (nats), max p, L1 to uniform, KL(p || uniform)."""
    q = np.clip(p, 1e-30, None)
    h = -(q * np.log(q)).sum(1)
    return dict(entropy=h, maxp=p.max(1), l1=np.abs(p - 1.0 / K).sum(1), kl=math.log(K) - h)


def structure_mean(p, structure):
    _, inv, counts = np.unique(structure, return_inverse=True, return_counts=True)
    smean = np.zeros((len(counts), K))
    np.add.at(smean, inv, p)
    return smean / counts[:, None], inv, counts


def router_summary(p, structure):
    """Summary of p [units, K]; structure [units] gives the structure of each unit (equal to arange for Global)."""
    s = unit_stats(p)
    out = dict(n_units=len(p))
    for name, v in s.items():
        out.update({f"{name}_mean": v.mean(), f"{name}_min": v.min(), f"{name}_median": np.median(v),
                    f"{name}_p95": np.percentile(v, 95), f"{name}_max": v.max()})
    out.update({f"p_mean_{k}": v for k, v in enumerate(p.mean(0))})
    out.update({f"p_std_{k}": v for k, v in enumerate(p.std(0))})
    out["p_std_mean"] = p.std(0).mean()
    smean, inv, counts = structure_mean(p, structure)
    dev = p - smean[inv]
    within_var = np.zeros((len(counts), K))
    np.add.at(within_var, inv, dev ** 2)
    out["struct_p_std_mean"] = smean.std(0).mean()                       # across structures, of the structure mean
    out["within_std_mean"] = np.sqrt(within_var / counts[:, None]).mean()  # inside a structure, mean over structures and K
    out["within_l1_mean"] = np.abs(dev).sum(1).mean()                     # L1 of an atom to its structure mean
    return out


# ---------- experts ----------

def expert_stats(w, w0=None):
    """w [K, out, in] of one MoLE layer (float64); w0 the same layer at step 0."""
    def pair(x):
        x = x.reshape(K, -1)
        n = np.linalg.norm(x, axis=1)
        c = (x / n[:, None]) @ (x / n[:, None]).T
        off = c[~np.eye(K, dtype=bool)]
        mean = x.mean(0)
        spread = math.sqrt(((x - mean) ** 2).sum(1).mean()) / np.linalg.norm(mean)
        return off, spread, n, mean, x - mean

    off, spread, n, mean, dev = pair(w)
    out = dict(cos_mean=off.mean(), cos_min=off.min(), cos_max=off.max(), spread=spread,
               norm_cv=n.std() / n.mean(), norm_mean=n.mean(), mean_norm=np.linalg.norm(mean),
               uniform_eff_over_expert=0.13 * K * np.linalg.norm(mean) / n.mean())
    if w0 is not None:
        uoff, uspread, un, umean, _ = pair(w - w0)
        dev0 = (w0 - w0.mean(0)).reshape(K, -1)
        out.update(upd_cos_mean=uoff.mean(), upd_cos_min=uoff.min(), upd_spread=uspread,
                   upd_norm_over_init_norm=un.mean() / np.linalg.norm(w0.reshape(K, -1), axis=1).mean(),
                   dev_over_dev0=np.linalg.norm(dev) / np.linalg.norm(dev0))
    return out


def mole_weights(model):
    from fairchem.core.models.uma.nn.mole import MOLE
    return {name: m.weights.detach().double().cpu().numpy() for name, m in model.named_modules() if isinstance(m, MOLE)}


# ---------- data and model ----------

def load_unit(path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))            # the Local-6 checkpoint names classes in training/ and local_routing/
    import torch
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    # the training settings: float32, Euler-angle Wigner (use_quaternion_wigner False in the configs), general backend;
    # no stress task (molecules have no cell)
    settings = InferenceSettings(base_precision_dtype=torch.float32, use_quaternion_wigner=None, execution_mode="general",
                                 auto_add_default_untrained_tasks=False)
    return pretrained_mlip.load_predict_unit(str(path), inference_settings=settings, device="cuda")


def load_data(src, rows, max_atoms):
    """Batches of AtomicData (at most max_atoms atoms each, in row order) and float64 targets."""
    import yaml
    from fairchem.core.datasets import AseDBDataset
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    refs = np.array(yaml.safe_load((ROOT / "configs/element_refs/omol.yaml").read_text(encoding="utf-8"))["omol_elem_refs"])
    db = AseDBDataset({"src": str(src)})
    batches, group, n = [], [], 0
    for row in rows:
        atoms = db.get_atoms(int(row))
        z = atoms.get_atomic_numbers()
        item = dict(row=int(row), natoms=len(atoms), resid=atoms.get_potential_energy() - refs[z].sum(),
                    forces=atoms.get_forces().astype(np.float64), refs=refs[z].sum(),
                    data=AtomicData.from_ase(atoms, task_name="omol", r_edges=False, r_data_keys=["spin", "charge"],
                                             molecule_cell_size=120.0))
        if group and n + item["natoms"] > max_atoms:
            batches.append(group)
            group, n = [], 0
        group.append(item)
        n += item["natoms"]
    if group:
        batches.append(group)
    return [(atomicdata_list_to_batch([i["data"] for i in g]), g) for g in batches]


def router_pass(unit, batches):
    import torch
    backbone = unit.model.module.backbone
    p, structure, offset = [], [], 0
    with torch.no_grad():
        for batch, items in batches:
            if not unit.lazy_model_intialized:
                unit._lazy_init(batch)
            data = batch.to(unit.device).clone()
            a = route(backbone, data).double().cpu().numpy()
            p.append(softmax_part(a))
            b = data["batch"].cpu().numpy() if hasattr(backbone, "routing_radius") else np.arange(len(items))
            structure.append(b + offset)
            offset += len(items)
    return np.concatenate(p), np.concatenate(structure)


def eval_pass(unit, batches, alpha=None, record=None):
    """Energy MAE per atom (eV) and force MAE (eV/A, per component) over all structures; per-structure errors."""
    backbone = unit.model.module.backbone
    e_err, f_abs, f_n = [], 0.0, 0
    with fixed_alpha(backbone, alpha, record):
        for batch, items in batches:
            out = unit.predict(batch, undo_element_references=False)
            e = out["energy"].detach().double().cpu().numpy().reshape(-1)
            f = out["forces"].detach().double().cpu().numpy()
            start = 0
            for i, item in enumerate(items):
                e_err.append(abs(e[i] - item["resid"]) / item["natoms"])
                d = f[start:start + item["natoms"]] - item["forces"]
                f_abs += np.abs(d).sum()
                f_n += d.size
                start += item["natoms"]
    return float(np.mean(e_err)), f_abs / f_n, np.array(e_err)


# ---------- run ----------

def checkpoints(run_dir):
    """{step: path}; 'final' is labelled with the run's total steps (as in size_consistency_along_training.py)."""
    steps = int(re.search(r"steps=(\d+)", (run_dir / "settings.txt").read_text()).group(1))
    out = {}
    for p in run_dir.glob("train/*/checkpoints/*/inference_ckpt.pt"):
        name = p.parent.name
        out[steps if name == "final" else int(name.removeprefix("step_"))] = p
    return out


def summary_metrics(run_dir):
    """The pilot's own last validation numbers (4 decimals) from summary.txt, for the reproduction check."""
    text = (run_dir / "summary.txt").read_text() if (run_dir / "summary.txt").exists() else ""
    e = re.findall(r"omol_energy,per_atom_mae: ([0-9.]+)", text)
    f = re.findall(r"omol_forces,mae: ([0-9.]+)", text)
    return (float(e[-1]) if e else float("nan")), (float(f[-1]) if f else float("nan"))


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


def run(a):
    import torch
    run_dir = Path(a.run_dir)
    m = re.match(r"^pilot_(global|local6)_s(\d+)_(\d+)$", run_dir.name)
    variant, seed = m.group(1), int(m.group(2))
    tag = dict(run=run_dir.name, variant=variant, seed=seed)
    ckpts = checkpoints(run_dir)
    final = max(ckpts)
    steps = sorted({s for s in [0, *a.steps] if s in ckpts} | {final})
    rows = np.loadtxt(a.rows, dtype=np.int64, ndmin=1)[: a.limit]
    t0 = time.time()
    batches = load_data(a.src, rows, a.max_atoms)
    print(f"{run_dir.name}: {len(rows)} structures in {len(batches)} batches, {sum(len(g) for _, g in batches)} items, "
          f"loaded in {time.time() - t0:.0f} s; steps {steps}", flush=True)
    router_rows, expert_rows, abl_rows, saved = [], [], [], {}
    w0 = None
    for step in steps:
        t = time.time()
        unit = load_unit(ckpts[step])
        p, structure = router_pass(unit, batches)
        router_rows.append(dict(**tag, step=step, unit="atom" if variant == "local6" else "structure",
                                **router_summary(p, structure)))
        saved[f"p_struct_{step}"] = structure_mean(p, structure)[0].astype(np.float32)
        w = mole_weights(unit.model)
        if step == 0:
            w0 = w
        for name, wk in w.items():
            expert_rows.append(dict(**tag, step=step, layer=name, **expert_stats(wk, None if w0 is None or step == 0 else w0[name])))
        print(f"step {step}: router and experts in {time.time() - t:.0f} s; entropy mean "
              f"{router_rows[-1]['entropy_mean']:.5f} (uniform {math.log(K):.5f}), {len(w)} MoLE layers", flush=True)
        if step == final:
            pe, pf = summary_metrics(run_dir)
            mean_alpha = p.mean(0) + 0.005
            fixed = {"trained": None, "uniform": np.full(K, 1.0 / K + 0.005), "eighth": np.full(K, 1.0 / K),
                     "mean": mean_alpha}
            per_structure = {}
            for cond in a.conditions:
                t = time.time()
                record = [] if cond == "trained" else None
                e, f, e_err = eval_pass(unit, batches, fixed[cond], record)
                per_structure[cond] = e_err
                if record is not None:              # the router used in the forward pass equals route()
                    used = softmax_part(torch.cat(record).double().cpu().numpy())
                    assert used.shape == p.shape and np.abs(used - p).max() < 1e-5, np.abs(used - p).max()
                abl_rows.append(dict(**tag, step=step, condition=cond, n_structures=len(e_err),
                                     n_atoms=sum(i["natoms"] for _, g in batches for i in g),
                                     energy_mae_per_atom=e, force_mae=f,
                                     alpha=" ".join(f"{x:.6f}" for x in (fixed[cond] if fixed[cond] is not None else [])),
                                     pilot_energy_mae_per_atom=pe, pilot_force_mae=pf, seconds=round(time.time() - t, 1)))
                print(f"  {cond}: energy MAE per atom {e:.6f} eV, force MAE {f:.6f} eV/A ({time.time() - t:.0f} s); "
                      f"pilot summary {pe}, {pf}; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                      flush=True)
            saved.update({f"err_{c}": v for c, v in per_structure.items()})
        del unit
        torch.cuda.empty_cache()
    out = Path(a.out_dir)
    stem = f"router_check_{variant}_s{seed}"
    np.savez(out / f"{stem}_per_structure.npz", rows=np.array([i["row"] for _, g in batches for i in g]), **saved)
    write(out / f"{stem}_router.csv", router_rows)
    write(out / f"{stem}_experts.csv", expert_rows)
    write(out / f"{stem}_ablation.csv", abl_rows)


# ---------- analyze ----------

def analyze(a):
    import pandas as pd
    from scipy import stats
    src = Path(a.in_dir)
    res = ROOT / "results"
    merged = {}
    for kind in ("router", "experts", "ablation"):
        files = sorted(src.glob(f"router_check_*_s*_{kind}.csv"))
        merged[kind] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        merged[kind].to_csv(res / f"router_check_{kind}.csv", index=False, float_format="%.8g")
        print(f"router_check_{kind}.csv: {len(merged[kind])} rows from {len(files)} files")
    abl = merged["ablation"]
    base = abl[abl.condition == "trained"].set_index("run")
    tests = []
    for cond in [c for c in CONDITIONS if c != "trained"]:
        d = abl[abl.condition == cond].set_index("run")
        for metric in ("energy_mae_per_atom", "force_mae"):
            delta = (d[metric] - base.loc[d.index, metric])
            d = d.assign(delta=delta.values, variant=d.variant)
            for variant in ("global", "local6"):
                x = d[d.variant == variant]
                t = stats.ttest_rel(x[metric], base.loc[x.index, metric]) if len(x) > 1 else None
                tests.append(dict(condition=cond, metric=metric, variant=variant, n=len(x),
                                  trained_mean=base.loc[x.index, metric].mean(), ablated_mean=x[metric].mean(),
                                  delta_mean=x.delta.mean(), delta_sd=x.delta.std(ddof=1),
                                  delta_rel_mean=(x.delta / base.loc[x.index, metric]).mean(),
                                  paired_t=t.statistic if t else float("nan"), paired_p=t.pvalue if t else float("nan")))
            g, l = d[d.variant == "global"].delta, d[d.variant == "local6"].delta
            w = stats.ttest_ind(g, l, equal_var=False)
            tests.append(dict(condition=cond, metric=metric, variant="global minus local6 (delta)", n=len(g) + len(l),
                              delta_mean=g.mean() - l.mean(), paired_t=w.statistic, paired_p=w.pvalue))
    pd.DataFrame(tests).to_csv(res / "router_check_tests.csv", index=False, float_format="%.6g")
    print(f"router_check_tests.csv: {len(tests)} rows")
    figure(merged, res / "figures" / "router_check.png")


def figure(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"global": "#C44E52", "local6": "#55A868"}
    names = {"global": "Global (per structure)", "local6": "Local-6 (per atom)"}
    r, abl = m["router"], m["ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, label in ((axes[0], "entropy_mean", "mean entropy of p (nats)"),
                           (axes[1], "l1_mean", "mean L1 distance of p from uniform")):
        for v, c in colors.items():
            dv = r[r.variant == v]
            for _, s in dv.groupby("seed"):
                ax.plot(s.step, s[col], color=c, lw=0.8, alpha=0.6, marker=".")
            mean = dv.groupby("step")[col].mean()
            ax.plot(mean.index, mean.values, color=c, lw=2.5, marker="o", label=f"{names[v]}, mean of 3 seeds")
        ax.set_xlabel("training step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].axhline(math.log(K), color="gray", ls="--", lw=1, label="uniform, ln 8")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    ax = axes[2]
    conds = [c for c in CONDITIONS if c != "trained"]
    base = abl[abl.condition == "trained"].set_index("run").force_mae
    for j, v in enumerate(colors):
        for run, s in abl[abl.variant == v].groupby("run"):
            s = s.set_index("condition").reindex(conds)
            xs = np.arange(len(conds)) + (j - 0.5) * 0.25
            ax.plot(xs, 100 * (s.force_mae.values / base[run] - 1), color=colors[v], lw=0.8, alpha=0.7, marker="o", ms=4)
        ax.plot([], [], color=colors[v], marker="o", label=names[v].split(" (")[0])
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel("force MAE change vs trained router (percent)")
    ax.set_title("ablation, final checkpoint, one line per seed", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--src", default="/ocean/projects/che260063p/shared/data/omol25/val")
    p.add_argument("--rows", default=str(ROOT / "manifests" / "pilot_val_5k.txt"))
    p.add_argument("--limit", type=int, default=None, help="first N rows only (local test)")
    p.add_argument("--steps", type=int, nargs="+", default=[0, 10000], help="router and expert checkpoints besides final")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--max-atoms", type=int, default=450)
    p.set_defaults(fn=run)
    p = sub.add_parser("analyze")
    p.add_argument("--in-dir", required=True)
    p.set_defaults(fn=analyze)
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.conditions[0] == "trained", "trained must come first"
    a.fn(a)


if __name__ == "__main__":
    main()
