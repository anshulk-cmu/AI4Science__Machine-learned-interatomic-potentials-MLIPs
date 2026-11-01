"""One loader per model family. load() returns (energy function in eV, graph cutoff in angstrom)."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

SPIN = {"omat": 0, "omol": 1}            # the default of each task; charge is 0 for both


def _energy(calc, info=None):
    """energy(atoms) in eV; energy.forces(atoms) in eV/angstrom."""
    def prepared(atoms):
        atoms = atoms.copy()
        if info:
            atoms.info.update(info)
        calc.reset()                         # ASE would return the cached result for identical atoms
        atoms.calc = calc
        return atoms

    def energy(atoms):
        return float(prepared(atoms).get_potential_energy())

    energy.forces = lambda atoms: prepared(atoms).get_forces()
    return energy


def _find(module, name):
    """First positive number stored under this attribute name in any submodule."""
    for m in module.modules():
        v = getattr(m, name, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    raise RuntimeError(f"no {name} found in the model")


def interaction_radius(energy, cutoff):
    """Finite graph radius only; Global routing still has no finite locality guarantee."""
    return max(cutoff, getattr(energy, "routing_radius", 0.0))


def _uma(model, task, dtype, version):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))        # a checkpoint of ours names classes in training/ and local_routing/
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    settings = "default"
    if dtype == "float64":
        settings = InferenceSettings(base_precision_dtype=torch.float64)
    if os.path.isfile(model):                # a checkpoint on disk, for example from our own training runs
        unit = pretrained_mlip.load_predict_unit(model, inference_settings=settings, device="cuda")
    else:
        unit = pretrained_mlip.get_predict_unit(model, inference_settings=settings, device="cuda")
    if version:                              # test only: override the stored router version
        for m in unit.model.modules():
            if hasattr(m, "model_version"):
                m.model_version = version
    calc = FAIRChemCalculator(unit, task_name=task)
    info = {"charge": 0, "spin": SPIN[task]}
    energy = _energy(calc, info)
    energy.unit = unit                       # lets a test reach the router
    energy.routing_radius = max((float(getattr(m, "routing_radius", 0.0))
                                 for m in unit.model.modules()), default=0.0)
    return energy, _find(unit.model, "cutoff")   # P1 confirms where the model lives


def _orb(dtype):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    model, adapter = pretrained.orb_v3_conservative_inf_omat(
        device="cuda", precision=dtype, compile=False)        # no Triton on Windows
    return _energy(ORBCalculator(model, adapter, device="cuda")), float(adapter.radius)


def _mace(model, dtype):
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, device="cuda", default_dtype=dtype)
    return _energy(calc), float(calc.models[0].r_max)


def load(family, model, task="notask", dtype="float64", version=None):
    if family == "uma":
        return _uma(model, task, dtype, version)
    if family == "orb":
        return _orb(dtype)
    return _mace(model, dtype)


def load_prediction_average(paths, task="omol", dtype="float64", device="cuda"):
    """Audit adapter for equal prediction averages; K models remain resident on the device."""
    import numpy as np
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from training.checkpoints import predictor, PredictionAverage, load_checkpoint, same_config
    paths = list(paths)
    if not paths:
        raise ValueError("at least one checkpoint is required")
    reference = load_checkpoint(paths[0])
    for path in paths[1:]:
        other = load_checkpoint(path)
        if not same_config(reference.model_config, other.model_config) or not same_config(reference.tasks_config, other.tasks_config):
            raise ValueError("ensemble model and task configurations must match")
    del reference
    units = [predictor(path, device, dtype) for path in paths]
    average = PredictionAverage(units)

    def predict(atoms):
        atoms = atoms.copy()
        atoms.info.update(charge=0, spin=SPIN[task])
        data = AtomicData.from_ase(atoms, task_name=task, r_edges=False,
                                  r_data_keys=["charge", "spin"], r_forces=False, r_stress=False,
                                  molecule_cell_size=120.0 if atoms.cell.volume == 0 else None, target_dtype=getattr(torch, dtype))
        return average.predict(atomicdata_list_to_batch([data]))

    def energy(atoms):
        return float(predict(atoms)["energy"].reshape(-1)[0])

    energy.forces = lambda atoms: np.asarray(predict(atoms)["forces"])
    energy.routing_radius = max(float(getattr(m, "routing_radius", 0.0)) for u in units for m in u.model.modules())
    return energy, max(_find(u.model, "cutoff") for u in units)
