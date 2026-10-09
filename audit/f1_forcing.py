"""F1: force the router to see the other fragment's composition and check that this alone reproduces Delta.

Delta = [E_A(alpha_AB) - E_A(alpha_A)] + [E_B(alpha_AB) - E_B(alpha_B)]. Each bracket is measured on one fragment
alone, with only the composition fed to the router swapped for the union's. Pass if the sum equals Delta within tau.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

import analyze
import geometry as g
import models

ROOT = Path(__file__).resolve().parents[1]
GAP = 100


def hook(unit):
    """Wrap the router so its composition input can be replaced. Returns the state dict that controls it."""
    backbone = next(m for m in unit.model.modules() if hasattr(m, "set_MOLE_coefficients"))
    original, state = backbone.set_MOLE_coefficients, {"z": None, "calls": 0}

    def patched(atomic_numbers_full, batch_full, csd_mixed_emb):
        if state["z"] is not None:
            state["calls"] += 1
            z = torch.as_tensor(state["z"], device=atomic_numbers_full.device, dtype=atomic_numbers_full.dtype)
            return original(z, torch.zeros_like(z), csd_mixed_emb)    # one system holding the union's atoms
        return original(atomic_numbers_full, batch_full, csd_mixed_emb)

    backbone.set_MOLE_coefficients = patched
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--model-version", type=float, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-balance", action="store_true")   # test only: switch off the charge balancing
    ap.add_argument("--tau-label", default=None)           # null tests to take tau from, if the label has none
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()

    energy, cutoff = models.load("uma", a.model, a.task, a.dtype, a.model_version)
    label = a.label or a.model + (f"-v{a.model_version:g}" if a.model_version else "")
    tau = analyze.tolerances()[(a.tau_label or label, a.task)]
    state = hook(energy.unit)
    if a.no_balance:
        # This changes the model, so the Delta it gives is not a result. Only the charge channels are switched off;
        # the spin channels are left untouched.
        for m in energy.unit.model.modules():
            if hasattr(m, "charge_channel_end"):
                m.charge_channel_end = m.charge_channel_start
    ideal = {s: g.ideal_slab(s) for s in ("Al", "Ni")}
    rows = []

    def forced(atoms, z):
        state["z"] = z
        try:
            return energy(atoms)
        finally:
            state["z"] = None

    for run in range(a.runs + 1):
        al = g.rattled(ideal["Al"], g.seed_for(run, "Al_a"))
        ni = g.rattled(ideal["Ni"], g.seed_for(run, "Ni_a"))
        pair = g.stacked(al, ni, GAP)
        g.assert_disconnected(pair, cutoff)
        z_union = pair.numbers
        e_al, e_ni, e_pair = energy(g.isolated(al)), energy(g.isolated(ni)), energy(pair)
        calls = state["calls"]
        part_al = forced(g.isolated(al), z_union) - e_al
        part_ni = forced(g.isolated(ni), z_union) - e_ni
        assert state["calls"] == calls + 2, "the router hook was not used"
        control = forced(g.isolated(al), g.isolated(al).numbers) - e_al      # own composition must change nothing
        assert abs(control) * 1000.0 <= tau, f"run {run}: control {control * 1000.0:.4g} meV is above tau"
        delta = (e_pair - e_al - e_ni) * 1000.0
        rows.append(dict(model=label, task=a.task, run=run, al_part_mev=part_al * 1000.0,
                         ni_part_mev=part_ni * 1000.0, delta_mev=delta,
                         residual_mev=(part_al + part_ni) * 1000.0 - delta, control_mev=control * 1000.0, tau_mev=tau))
        print(f"{label} {a.task} run {run}: delta {delta:.3f}, forced sum {(part_al + part_ni) * 1000.0:.3f} meV")

    out = ROOT / "results" / f"f1_{label}_{a.task}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    worst = df.residual_mev.abs().max()
    print(f"max |residual| {worst:.4g} meV, max |control| {df.control_mev.abs().max():.4g} meV, tau {tau:.4g} meV")
    print("F1", "PASS" if worst <= tau else "FAIL", "(max |residual| <= tau)")


if __name__ == "__main__":
    main()
