"""Slabs, the shared strained cell, the stacked pair, and the checks on them."""
import numpy as np
from ase.build import fcc111
from ase.neighborlist import neighbor_list

LATTICE = {"Al": 4.05, "Ni": 3.52}      # ASE reference states, angstrom
SIZE = (4, 4, 8)                         # 4x4 in plane, 8 layers
SIGMA = 0.05                             # rattle, angstrom per Cartesian component
PAD = 20.0                               # empty space above an isolated slab
DISTANCES = (8, 10, 15, 25, 50, 100)     # surface to surface gap, angstrom
FRAGMENT_ID = {"Al_a": 1, "Al_b": 2, "Ni_a": 3, "Ni_b": 4}


def spacing(symbol):
    """In-plane nearest-neighbour spacing of fcc(111)."""
    return LATTICE[symbol] / np.sqrt(2)


SHARED = (spacing("Al") + spacing("Ni")) / 2


def ideal_slab(symbol):
    """Unrelaxed slab with the in-plane cell strained to the shared spacing, bottom plane at z = 0."""
    slab = fcc111(symbol, size=SIZE, a=LATTICE[symbol], vacuum=None)
    scale = SHARED / spacing(symbol)
    cell = slab.cell.array.copy()
    cell[:2] *= scale
    slab.positions[:, :2] *= scale
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    slab.set_cell(cell)
    slab.info["thickness"] = float(slab.positions[:, 2].max())
    slab.info["strain"] = scale - 1.0
    return slab


def seed_for(run, name):
    """Seed of one fragment in one run. Run 0 is the ideal geometry."""
    return 0 if run == 0 else 1000 * run + FRAGMENT_ID[name]


def rattled(slab, seed):
    out = slab.copy()
    if seed:
        out.positions += np.random.default_rng(seed).normal(0.0, SIGMA, out.positions.shape)
    return out


def isolated(slab, pad=PAD):
    """The slab alone, periodic in all directions, with empty space above it."""
    out = slab.copy()
    cell = out.cell.array.copy()
    cell[2] = [0.0, 0.0, slab.info["thickness"] + pad]
    out.set_cell(cell)
    out.pbc = True
    return out


def stacked(lower, upper, gap):
    """Upper slab above lower. The gap is the same on both sides of each slab in the periodic cell."""
    t_low, t_up = lower.info["thickness"], upper.info["thickness"]
    top = upper.copy()
    top.positions[:, 2] += t_low + gap
    pair = lower + top
    cell = lower.cell.array.copy()
    cell[2] = [0.0, 0.0, t_low + gap + t_up + gap]
    pair.set_cell(cell)
    pair.pbc = True
    pair.info = {"n_lower": len(lower)}
    return pair


def disconnected(pair, cutoff, margin=1.0):
    """Check all cross-fragment pairs, including periodic images, with a safety margin."""
    if not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("cutoff must be positive and margin nonnegative, both finite")
    i, j = neighbor_list("ij", pair, cutoff + margin)
    n = pair.info["n_lower"]
    return not bool(((i < n) != (j < n)).any())


def assert_disconnected(pair, cutoff, margin=1.0):
    """Stop if any atom of one fragment is near the other (periodic images included)."""
    if not disconnected(pair, cutoff, margin):
        raise RuntimeError("fragments are connected within the model cutoff")


def selftest():
    al, ni = ideal_slab("Al"), ideal_slab("Ni")
    assert len(al) == len(ni) == 128
    assert np.allclose(al.cell.array[:2], ni.cell.array[:2], atol=1e-9)
    assert abs(al.info["strain"] + 0.0654) < 1e-4 and abs(ni.info["strain"] - 0.0753) < 1e-4
    for slab, sym in ((al, "Al"), (ni, "Ni")):
        layers = np.unique(slab.positions[:, 2].round(6))
        assert len(layers) == 8
        assert np.allclose(np.diff(layers), LATTICE[sym] / np.sqrt(3), atol=1e-4)
    for d in DISTANCES:
        pair = stacked(al, ni, d)
        assert_disconnected(pair, 6.0)
        assert np.isclose(pair.cell[2, 2], al.info["thickness"] + ni.info["thickness"] + 2 * d)
    a = rattled(al, seed_for(1, "Al_a"))
    assert np.array_equal(a.positions, rattled(al, seed_for(1, "Al_a")).positions)
    assert abs((a.positions - al.positions).std() - SIGMA) < 0.1 * SIGMA
    print("geometry selftest passed")


if __name__ == "__main__":
    selftest()
