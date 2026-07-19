"""Shared helpers for the oxDNA plugin tests.

Centralises the reference-data access, snapshot/simulation setup, quaternion
utilities, and the finite-difference force/torque harness that every test file
would otherwise re-implement.
"""

import os
import numpy as np
import hoomd

from oxdna import io

REF_DIR = os.environ.get(
    "OXDNA_TESTDATA",
    "/groups/goloborodko/user/anton.goloborodko/src/jax-dna/jax-dna/data/test-data",
)


def make_device():
    """CPU by default; GPU when OXDNA_TEST_DEVICE=gpu (runs the same suite on GPU)."""
    if os.environ.get("OXDNA_TEST_DEVICE", "cpu").lower() == "gpu":
        return hoomd.device.GPU()
    return hoomd.device.CPU()

# split_energy.dat column index (0 = step) for each per-nucleotide energy term.
COL = {
    "fene": 1, "b_exc": 2, "stack": 3, "n_exc": 4,
    "hb": 5, "cr_stack": 6, "cx_stack": 7, "debye": 8,
}


def case_paths(case, top_name="generated.top", conf_name="start.conf"):
    """(top, conf) paths for a case. DNA cases use generated.top/start.conf; the
    oxRNA reference cases use sys.top/init.conf (pass those names)."""
    d = os.path.join(REF_DIR, case)
    return os.path.join(d, top_name), os.path.join(d, conf_name)


def split_energy(case, frame=0):
    """Reference per-nucleotide energies for a config frame (row of split_energy.dat)."""
    return np.loadtxt(os.path.join(REF_DIR, case, "split_energy.dat"))[frame]


def snapshot(case, half_charged_ends=False, top_name="generated.top", conf_name="start.conf",
             frame=0):
    top, conf = case_paths(case, top_name, conf_name)
    return io.build_snapshot(top, conf, frame=frame, half_charged_ends=half_charged_ends)


def run_zero(snap, forces):
    """Create a simulation, attach forces, and advance zero steps (single-point).

    The returned simulation MUST be kept alive while reading force energies/outputs
    (dropping it detaches the forces).
    """
    sim = hoomd.Simulation(device=make_device())
    sim.create_state_from_snapshot(snap)
    integ = hoomd.md.Integrator(dt=0.0, integrate_rotational_dof=True, forces=list(forces))
    sim.operations.integrator = integ
    sim.run(0)
    return sim


def single_point(snap, forces, read):
    """Attach forces, run 0 steps, and return ``read(forces)`` while the sim is alive."""
    sim = run_zero(snap, forces)
    result = read(forces)
    del sim
    return result


def run_langevin(snap, forces, kT, steps, seed=1, dt=0.003):
    """Run a Langevin trajectory (used by the duplex-stability tests)."""
    sim = hoomd.Simulation(device=make_device(), seed=seed)
    sim.create_state_from_snapshot(snap)
    lang = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
    integ = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True,
                                forces=list(forces), methods=[lang])
    sim.operations.integrator = integ
    sim.run(steps)
    return sim


def load_frame(case, frame=0, top_name="generated.top", conf_name="start.conf"):
    """Return a dict of arrays (pos, quats, typeids, box, bonds, charges) for a frame.

    ``base_typeid`` maps U to T's type index, so oxRNA cases work unchanged.
    """
    top_path, conf_path = case_paths(case, top_name, conf_name)
    top = io.read_topology(top_path)
    _, box, arr = list(io.iter_conf_frames(conf_path))[frame]
    return dict(
        pos=arr[:, 0:3] - box / 2.0,
        quats=io.orientations_to_quaternions(arr[:, 3:6], arr[:, 6:9]),
        typeids=np.array([io.base_typeid(b) for b in top["base"]]),
        box=box,
        bonds=[(i, int(top["n5"][i])) for i in range(top["n"]) if top["n5"][i] >= 0],
        charges=np.ones(top["n"]),
    )


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def delta_quat(axis, eps):
    return np.array([np.cos(eps / 2.0), *(np.sin(eps / 2.0) * np.asarray(axis))])


def measure(build_forces, frame, pos=None, quats=None):
    """Attach ``build_forces()`` to a snapshot from ``frame`` and read force[0].

    ``build_forces`` is a zero-arg factory returning a list of forces (index 0 is
    the term under test); it must build a fresh neighbor list each call. ``pos`` /
    ``quats`` override the frame arrays (used by the finite-difference sweeps).
    """
    p = frame["pos"] if pos is None else pos
    q = frame["quats"] if quats is None else quats
    snap = io.snapshot_from_arrays(p, q, frame["typeids"], frame["box"],
                                   bonds=frame["bonds"], charges=frame["charges"])
    forces = build_forces()
    sim = run_zero(snap, forces)  # keep the sim alive while reading the outputs below
    f = forces[0]
    result = float(f.energy), np.array(f.forces), np.array(f.torques)
    del sim
    return result


def fd_check(build_forces, frame, rtol=1e-3, atol=5e-3):
    """Finite-difference the force and torque of ``build_forces()[0]`` on ``frame``.

    Perturbs the most-active particle's position and orientation and compares the
    central differences of the energy with the reported force/torque. Also checks
    Newton's third law. Tolerances are relative+absolute (float32 fork noise).

    Step size: the nonbonded evaluators compute in single precision on the
    mixed-precision fork, so the energies fed to the central difference carry ~1e-7
    relative noise. Central-difference error is truncation (~step^2) plus roundoff
    (~eps_float*|E|/step); the optimum sits near eps_float**(1/3) ~ 4e-3 for float,
    versus ~5e-6 for double. A 1e-4 step is roundoff-dominated for a stiff term
    (excluded volume energies ~30 give ~eps*E/step ~ 0.3 of FD noise), so we use a
    float-appropriate 2e-3 step: truncation stays ~1e-5 while roundoff drops ~20x.
    """
    _, forces, torques = measure(build_forces, frame)
    # Sum(forces) should vanish. In single precision (GPU fork) the roundoff of that
    # cancellation scales with the force magnitude, so the tolerance must too: a real
    # third-law violation is O(force), far above this float-noise floor.
    fmax = float(np.abs(forces).max())
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-5 + 1e-5 * fmax, "Newton's third law violated"
    i = int(np.argmax(np.linalg.norm(forces, axis=1)))

    h = 2e-3
    for d in range(3):
        pp = frame["pos"].copy(); pp[i, d] += h
        ep = measure(build_forces, frame, pos=pp)[0]
        pp[i, d] -= 2 * h
        em = measure(build_forces, frame, pos=pp)[0]
        f_fd = -(ep - em) / (2 * h)
        assert abs(f_fd - forces[i, d]) < rtol * abs(forces[i, d]) + atol, (
            f"force[{i},{d}] analytic={forces[i, d]:.6f} fd={f_fd:.6f}"
        )

    eps = 2e-3
    for k, axis in enumerate(np.eye(3)):
        qp = frame["quats"].copy(); qp[i] = quat_mul(delta_quat(axis, eps), frame["quats"][i])
        ep = measure(build_forces, frame, quats=qp)[0]
        qm = frame["quats"].copy(); qm[i] = quat_mul(delta_quat(axis, -eps), frame["quats"][i])
        em = measure(build_forces, frame, quats=qm)[0]
        t_fd = -(ep - em) / (2 * eps)
        assert abs(t_fd - torques[i, k]) < rtol * abs(torques[i, k]) + atol, (
            f"torque[{i},{k}] analytic={torques[i, k]:.6f} fd={t_fd:.6f}"
        )
    return i
