"""Read oxDNA topology (.top) and configuration (.conf/.dat) into HOOMD.

oxDNA represents each nucleotide as a rigid body: a centre of mass plus the two
body axes a1 (backbone->base) and a3 (base normal) written in the configuration
file. HOOMD represents orientation as a quaternion, so we build the body-frame
rotation matrix ``R = [a1 | a2 | a3]`` (with ``a2 = a3 x a1``) and convert it to a
(w, x, y, z) quaternion. HOOMD's ``rotate(q, e_x) == a1`` by construction, which is
the convention every oxDNA force evaluator relies on.

Units are oxDNA simulation units throughout (no conversion).
"""

import numpy as np
import hoomd

#: oxDNA base alphabet -> HOOMD particle types (index is the HOOMD typeid).
#: Order matches oxDNA's encoding (defs.h: N_A=0, N_G=1, N_C=2, N_T=3), so a
#: HOOMD typeid equals the oxDNA base index. Complementary pairs sum to 3
#: (A+T, G+C), which the hydrogen-bonding term relies on.
DNA_BASES = ["A", "G", "C", "T"]

#: RNA topologies spell the fourth base U (uracil); it shares T's type index, so the
#: same A+X=3 Watson-Crick complementarity and the same 4 HOOMD types are used.
BASE_ALIAS = {"U": "T"}


def base_typeid(letter):
    """HOOMD typeid for an oxDNA/oxRNA base letter (U is treated as T)."""
    return DNA_BASES.index(BASE_ALIAS.get(letter, letter))


def read_topology(top_path):
    """Parse an oxDNA .top file.

    Returns a dict with ``n`` (nucleotide count), ``n_strands``, ``strand``
    (per-nucleotide strand id), ``base`` (list of base letters), and ``n3``/``n5``
    (3' and 5' neighbour indices, -1 if none).
    """
    with open(top_path) as fh:
        header = fh.readline().split()
        n, n_strands = int(header[0]), int(header[1])
        strand = np.empty(n, dtype=int)
        base = []
        n3 = np.empty(n, dtype=int)
        n5 = np.empty(n, dtype=int)
        for i in range(n):
            parts = fh.readline().split()
            strand[i] = int(parts[0])
            base.append(parts[1])
            n3[i] = int(parts[2])
            n5[i] = int(parts[3])
    return dict(n=n, n_strands=n_strands, strand=strand, base=base, n3=n3, n5=n5)


def iter_conf_frames(conf_path):
    """Yield ``(t, box, arr)`` for each frame in an oxDNA conf/trajectory file.

    ``arr`` has one row per nucleotide with 15 columns:
    ``com(3), a1(3), a3(3), velocity(3), angular_velocity(3)``.
    """
    with open(conf_path) as fh:
        lines = fh.readlines()
    i, nlines = 0, len(lines)
    while i < nlines:
        if not lines[i].startswith("t ="):
            i += 1
            continue
        t = float(lines[i].split("=")[1])
        box = np.array(lines[i + 1].split("=")[1].split(), dtype=float)
        i += 3  # skip 't =', 'b =', 'E =' lines
        rows = []
        while i < nlines and not lines[i].startswith("t ="):
            s = lines[i].split()
            if len(s) >= 9:
                rows.append([float(x) for x in s[:15]])
            i += 1
        yield t, box, np.array(rows, dtype=float)


def _matrix_to_quaternion(R):
    """Rotation matrix (columns a1, a2, a3) -> (w, x, y, z) unit quaternion.

    Shepperd's numerically-stable branch selection.
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def orientations_to_quaternions(a1, a3, check=True):
    """Convert per-nucleotide (a1, a3) axes to (w, x, y, z) quaternions."""
    n = a1.shape[0]
    quats = np.empty((n, 4), dtype=float)
    for i in range(n):
        e1 = a1[i] / np.linalg.norm(a1[i])
        e3 = a3[i] / np.linalg.norm(a3[i])
        e2 = np.cross(e3, e1)  # a2 = a3 x a1
        R = np.column_stack([e1, e2, e3])
        q = _matrix_to_quaternion(R)
        if check:
            # verify rotate(q, e_x) reproduces a1 (guards the quaternion convention)
            recon = _rotate(q, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(recon - e1) > 1e-6:
                raise ValueError(
                    f"quaternion convention check failed at nucleotide {i}: "
                    f"|rotate(q,ex) - a1| = {np.linalg.norm(recon - e1):.2e}"
                )
        quats[i] = q
    return quats


def _rotate(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z) (matches HOOMD's rotate)."""
    w, x, y, z = q
    # R @ v with R built from q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ v


def snapshot_from_arrays(positions, quaternions, typeids, box, bonds=None,
                         types=DNA_BASES, bond_type="backbone", charges=None):
    """Build a HOOMD snapshot directly from arrays (positions already box-centred).

    ``bonds`` is an optional list of (i, j) index pairs. ``charges`` is an optional
    per-particle charge array (used for oxDNA2 half-charged strand ends).
    """
    n = len(positions)
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.particles.N = n
        snap.particles.types = list(types)
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = typeids
        snap.particles.mass[:] = 1.0
        snap.particles.moment_inertia[:] = np.ones((n, 3))
        snap.particles.orientation[:] = quaternions
        if charges is not None:
            snap.particles.charge[:] = charges
        snap.configuration.box = [box[0], box[1], box[2], 0, 0, 0]
        bonds = bonds or []
        snap.bonds.N = len(bonds)
        snap.bonds.types = [bond_type]
        if bonds:
            snap.bonds.group[:] = np.array(bonds, dtype=int)
            snap.bonds.typeid[:] = np.zeros(len(bonds), dtype=int)
    return snap


def terminal_charges(top, end_charge=0.5, interior_charge=1.0):
    """Per-nucleotide charge for oxDNA2: half at strand ends (n3 or n5 missing)."""
    n = top["n"]
    charges = np.full(n, interior_charge)
    ends = (top["n3"] < 0) | (top["n5"] < 0)
    charges[ends] = end_charge
    return charges


def build_snapshot(top_path, conf_path, frame=0, bond_type="backbone",
                   half_charged_ends=False):
    """Build a HOOMD snapshot from an oxDNA topology + configuration frame.

    Backbone bonds are created from the 5' neighbour list (each bond once).
    Particle types are the DNA bases; positions are recentred into the HOOMD box.
    """
    top = read_topology(top_path)
    n = top["n"]
    frames = list(iter_conf_frames(conf_path))
    _, box, arr = frames[frame]

    pos = arr[:, 0:3].copy()
    a1 = arr[:, 3:6].copy()
    a3 = arr[:, 6:9].copy()

    # HOOMD box is centred at the origin; oxDNA coords are in [0, L].
    pos_centered = pos - box / 2.0

    quats = orientations_to_quaternions(a1, a3)
    typeids = np.array([base_typeid(b) for b in top["base"]], dtype=int)

    # Backbone bonds: one per 5' link.
    n5 = top["n5"]
    bonds = [(i, int(n5[i])) for i in range(n) if n5[i] >= 0]

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.particles.N = n
        snap.particles.types = list(DNA_BASES)
        snap.particles.position[:] = pos_centered
        snap.particles.typeid[:] = typeids
        snap.particles.mass[:] = 1.0
        snap.particles.moment_inertia[:] = np.ones((n, 3))
        snap.particles.orientation[:] = quats
        snap.particles.charge[:] = (
            terminal_charges(top) if half_charged_ends else np.ones(n)
        )
        snap.configuration.box = [box[0], box[1], box[2], 0, 0, 0]

        snap.bonds.N = len(bonds)
        snap.bonds.types = [bond_type]
        if bonds:
            snap.bonds.group[:] = np.array(bonds, dtype=int)
            snap.bonds.typeid[:] = np.zeros(len(bonds), dtype=int)
    return snap
