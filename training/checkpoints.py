"""Select and combine inference checkpoints from one training run."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    rows = [int(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or min(rows) < 0 or len(set(rows)) != len(rows):
        raise ValueError("row lists must be nonempty, unique and nonnegative")
    return rows


def split_rows(selection_path, report_path):
    selection, report = read_rows(selection_path), read_rows(report_path)
    if set(selection) & set(report):
        raise ValueError("selection and report rows overlap")
    return selection, report


def discover(run_dir, total_steps):
    """One trainer directory per run. Numeric folders and final may duplicate the last step."""
    run_dir = Path(run_dir).resolve()
    roots = list(run_dir.glob("train/*/checkpoints"))
    if len(roots) != 1 or total_steps <= 0:
        raise ValueError("need one checkpoint directory and a positive total step count")
    found = {}
    for path in sorted(roots[0].glob("*/inference_ckpt.pt")):
        folder = path.parent.name
        if folder == "final":
            # A failed or interrupted run must not silently acquire the planned step count.
            summary = (run_dir / "summary.txt").read_text()
            if "RESULT: all stages passed" not in summary:
                raise ValueError("final checkpoint needs a successful run summary; use numeric step folders for partial runs")
            step = total_steps
        elif re.fullmatch(r"(?:step_)?\d+", folder):
            step = int(folder.removeprefix("step_"))
        else:
            continue
        if not 0 <= step <= total_steps:
            raise ValueError(f"checkpoint step outside this run: {path}")
        # Prefer final over a duplicate numeric checkpoint at the same step.
        if step not in found or folder == "final":
            found[step] = path.resolve()
    if not found:
        raise ValueError("no inference checkpoints found")
    if max(found) != total_steps:
        raise ValueError("last saved step does not equal the supplied completed run length")
    return sorted(found.items())


def select(rows, total_steps, k=5):
    if k < 1 or total_steps <= 0:
        raise ValueError("k and total_steps must be positive")
    if len({r["step"] for r in rows}) != len(rows):
        raise ValueError("duplicate checkpoint steps")
    candidates = []
    for row in rows:
        values = [row["energy_mae_per_atom_eV"], row["force_mae_eV_per_A"]]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("metrics must be finite and nonnegative")
        if total_steps / 2 <= row["step"] <= total_steps:
            candidates.append(row)
    if len(candidates) < k:
        raise ValueError(f"need {k} checkpoints in the last half; found {len(candidates)}")
    return sorted(candidates, key=lambda r: (r["force_mae_eV_per_A"], r["step"]))[:k]


def load_checkpoint(path):
    # Inference checkpoints contain a fairchem dataclass. Only load trusted training outputs.
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, MLIPInferenceCheckpoint) or not checkpoint.ema_state_dict:
        raise ValueError("expected a fairchem inference checkpoint with EMA weights")
    return checkpoint


def same_config(a, b):
    from omegaconf import OmegaConf

    def plain(value):
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    return json.dumps(plain(a), sort_keys=True) == json.dumps(plain(b), sort_keys=True)


def average_weights(paths, output):
    """Mean EMA parameters, unchanged buffers, inference-only output (no optimizer state)."""
    import hydra
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct checkpoint paths")
    if Path(output).exists():
        raise FileExistsError(output)
    first = load_checkpoint(paths[0])
    model = hydra.utils.instantiate(first.model_config)
    parameter_names = {"module." + name for name, _ in model.named_parameters()}
    del model
    state = first.ema_state_dict
    if not parameter_names <= state.keys():
        raise ValueError("EMA parameter names do not match the model")
    sums = {key: value.to(torch.float64).clone() for key, value in state.items() if key in parameter_names}
    for path in paths[1:]:
        checkpoint = load_checkpoint(path)
        if not same_config(first.model_config, checkpoint.model_config) or not same_config(first.tasks_config, checkpoint.tasks_config):
            raise ValueError("model and task configurations must match, including normalizers and element references")
        other = checkpoint.ema_state_dict
        if state.keys() != other.keys():
            raise ValueError("EMA state keys differ")
        for key, value in other.items():
            reference = state[key]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"incompatible tensor: {key}")
            if key in parameter_names:
                sums[key].add_(value.to(torch.float64))
            elif key != "n_averaged" and not torch.equal(value, reference):
                raise ValueError(f"non-parameter buffer differs: {key}")
    averaged = copy.deepcopy(state)
    for key, value in sums.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite averaged parameter: {key}")
        averaged[key] = (value / len(paths)).to(state[key].dtype)
    # This counter describes EMA updates, not ensemble membership. Mark the new model initialized.
    averaged["n_averaged"] = torch.ones_like(state["n_averaged"])
    raw = {key.removeprefix("module."): value.clone() for key, value in averaged.items() if key != "n_averaged"}
    result = MLIPInferenceCheckpoint(copy.deepcopy(first.model_config), raw, averaged, copy.deepcopy(first.tasks_config))
    with open(output, "xb") as stream:
        torch.save(result, stream)
    return result


def predictor(path, device="cpu", dtype="float64"):
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    settings = InferenceSettings(base_precision_dtype=getattr(torch, dtype), use_quaternion_wigner=None,
                                 execution_mode="general", auto_add_default_untrained_tasks=False,
                                 merge_mole=False)
    return MLIPPredictUnit(str(path), device=device, inference_settings=settings)


class PredictionAverage:
    """Equal, position-independent weights preserve energy/force consistency."""
    def __init__(self, units):
        if not units:
            raise ValueError("at least one prediction unit is required")
        self.units = units

    def predict(self, data):
        total = None
        for unit in self.units:
            values = {key: value.detach().cpu().double() for key, value in unit.predict(data.clone()).items()}
            if total is None:
                total = values
            else:
                if total.keys() != values.keys():
                    raise ValueError("prediction task keys differ")
                for key, value in values.items():
                    if value.shape != total[key].shape:
                        raise ValueError(f"prediction shape differs: {key}")
                    total[key].add_(value)
        return {key: value / len(self.units) for key, value in total.items()}


def evaluate(unit, dataset, rows):
    """Energy: mean per-structure |dE|/N. Force: mean over all Cartesian components."""
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    energy_error = force_error = 0.0
    components = 0
    for index in rows:
        atoms = dataset.get_atoms(index)
        if not len(atoms) or not {"charge", "spin"} <= atoms.info.keys():
            raise ValueError(f"row {index} needs atoms, charge and spin")
        target_e, target_f = atoms.get_potential_energy(), atoms.get_forces()
        data = AtomicData.from_ase(atoms, task_name="omol", r_edges=False,
                                  r_data_keys=["charge", "spin"], r_forces=False, r_stress=False,
                                  molecule_cell_size=120.0 if atoms.cell.volume == 0 else None, target_dtype=torch.float64)
        pred = unit.predict(atomicdata_list_to_batch([data]))
        e = float(pred["energy"].detach().cpu().reshape(-1)[0])
        f = pred["forces"].detach().cpu().numpy()
        if f.shape != target_f.shape or not np.isfinite(f).all() or not np.isfinite(target_f).all() or not math.isfinite(e + target_e):
            raise ValueError(f"nonfinite or malformed prediction/target at row {index}")
        energy_error += abs(e - target_e) / len(atoms)
        force_error += np.abs(f - target_f).sum()
        components += target_f.size
    if not rows:
        raise ValueError("evaluation set is empty")
    return dict(energy_mae_per_atom_eV=energy_error / len(rows),
                force_mae_eV_per_A=float(force_error / components), n_structures=len(rows), n_force_components=components)


def checked_selection(path):
    record = json.loads(Path(path).read_text())
    root = Path(record["run_dir"]).resolve()
    for row in record["selected"] + [record["final"]]:
        checkpoint = Path(row["path"]).resolve()
        if not checkpoint.is_relative_to(root) or sha256(checkpoint) != row["sha256"]:
            raise ValueError("checkpoint moved outside its run or changed since selection")
    return record
