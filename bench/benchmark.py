"""CPU vs GPU benchmark for the oxDNA plugin.

Tiles the 16-nucleotide simple-helix duplex into a grid to make a large system,
then times a Langevin trajectory on the requested device.

    python bench/benchmark.py --device gpu --copies 12 --steps 500
    python bench/benchmark.py --device cpu --copies 6  --steps 200
"""

import argparse
import time
import numpy as np
import hoomd

import oxdna  # noqa: F401
from oxdna import io
from oxdna.model import dna1, dna2

REF = ("/groups/goloborodko/user/anton.goloborodko/src/jax-dna/jax-dna/"
       "data/test-data/simple-helix")


def tiled_snapshot(copies, spacing=8.0):
    """Replicate the simple-helix duplex on a copies^3 grid."""
    top = io.read_topology(f"{REF}/generated.top")
    _, _, arr = next(io.iter_conf_frames(f"{REF}/start.conf"))
    base_pos = arr[:, 0:3] - arr[:, 0:3].mean(axis=0)  # centre the duplex
    quats = io.orientations_to_quaternions(arr[:, 3:6], arr[:, 6:9])
    typeids = np.array([io.DNA_BASES.index(b) for b in top["base"]])
    n0 = top["n"]
    bonds0 = [(i, int(top["n5"][i])) for i in range(n0) if top["n5"][i] >= 0]

    L = copies * spacing
    positions, all_quats, all_types, all_bonds = [], [], [], []
    idx = 0
    for ix in range(copies):
        for iy in range(copies):
            for iz in range(copies):
                shift = (np.array([ix, iy, iz]) + 0.5) * spacing - L / 2.0
                positions.append(base_pos + shift)
                all_quats.append(quats)
                all_types.append(typeids)
                all_bonds += [(a + idx, b + idx) for (a, b) in bonds0]
                idx += n0
    positions = np.concatenate(positions)
    all_quats = np.concatenate(all_quats)
    all_types = np.concatenate(all_types)
    box = np.array([L, L, L])
    charges = np.ones(len(positions))
    snap = io.snapshot_from_arrays(positions, all_quats, all_types, box,
                                   bonds=all_bonds, charges=charges)
    return snap, len(positions)


def run(device_name, copies, steps, model, fused=False):
    dev = hoomd.device.GPU() if device_name == "gpu" else hoomd.device.CPU()
    snap, n = tiled_snapshot(copies)
    sim = hoomd.Simulation(device=dev, seed=1)
    sim.create_state_from_snapshot(snap)
    mod = dna2 if model == "dna2" else dna1
    forces, nlist = mod.forces(fused=fused)
    kT = dna1.kT_from_temperature(296.15)
    lang = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
    integ = hoomd.md.Integrator(dt=0.003, integrate_rotational_dof=True,
                                forces=forces, methods=[lang])
    sim.operations.integrator = integ

    sim.run(20)  # warm up (JIT, autotuner, neighbor list build)
    sim.run(0)
    t0 = time.perf_counter()
    sim.run(steps)
    dt = time.perf_counter() - t0
    tps = steps / dt
    return dict(device=device_name, n=n, steps=steps, seconds=dt, tps=tps,
                nuc_steps_per_s=n * tps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--model", choices=["dna1", "dna2"], default="dna2")
    ap.add_argument("--fused", action="store_true",
                    help="fuse hydrogen bonding + cross-stacking into one kernel")
    args = ap.parse_args()
    r = run(args.device, args.copies, args.steps, args.model, fused=args.fused)
    print(f"{r['device'].upper():3s}  N={r['n']:6d}  {r['steps']} steps in "
          f"{r['seconds']:.2f}s  =>  {r['tps']:.1f} steps/s  "
          f"({r['nuc_steps_per_s']:.3e} nucleotide-steps/s)")
