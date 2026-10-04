"""One loader per model family. load() returns (energy function in eV, graph cutoff in angstrom)."""
import os

import torch

SPIN = {"omat": 0, "omol": 1}            # the default of each task; charge is 0 for both


def _energy(calc, info=None):
    def energy(atoms):
        atoms = atoms.copy()
        if info:
            atoms.info.update(info)
        atoms.calc = calc
        return float(atoms.get_potential_energy())
    return energy


def _find(module, name):
    """First positive number stored under this attribute name in any submodule."""
    for m in module.modules():
        v = getattr(m, name, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    raise RuntimeError(f"no {name} found in the model")


def _uma(model, task, dtype, version):
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
    return _energy(calc, info), _find(unit.model, "cutoff")   # P1 confirms where the model lives


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
