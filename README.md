# hoomd-oxDNA

The oxDNA coarse-grained DNA model (oxDNA1 and oxDNA2) implemented as force
evaluators for [HOOMD-blue](https://hoomd-blue.readthedocs.io). Each nucleotide is
a single anisotropic rigid body (centre of mass + orientation quaternion) with
three interaction sites (BACK, BASE, STACK); the potential is the standard oxDNA
sum of FENE backbone, excluded volume, stacking, hydrogen bonding, cross-stacking,
coaxial stacking, and — for oxDNA2 — Debye-Hückel electrostatics.

## Layout

The project is split into a compiled engine and a pure-Python model layer.

```
src/                       C++ force engine (compiled into oxdna/_engine)
  OxDNANumerics.h            f1..f6 modulation factors, FENE, LJ core (host+device)
  OxDNAAnisoPair.h           CRTP base for the anisotropic pair evaluators
  OxDNAParamIO.h             pybind <-> POD parameter helpers
  EvaluatorPairOxDNA*.h      one evaluator per nonbonded term (excl. volume, H-bond,
                             cross-stacking, coaxial stacking, Debye-Hückel)
  OxDNABondedForceCompute.*  orientation-aware bonded force (FENE + bonded excl. vol
                             + stacking); these act at off-COM sites and need torque
  AnisoPotentialPairOxDNA*.cc  one-line registrations, module.cc  pybind module
oxdna/                     pure-Python package (no recompile to change a model)
  io.py                      .top/.conf <-> hoomd.Snapshot (quaternions, charges)
  forces.py                  thin hoomd.md wrappers over the engine
  model/dna1.py, dna2.py     geometry, parameters, and force-field assembly
pytest/                    validation against the oxDNA reference (split_energy.dat)
  _helpers.py                shared snapshot / finite-difference test harness
```

Why two force classes: HOOMD's central `PotentialBond` only applies a force along
the COM–COM axis with no torque. oxDNA's bonded terms act at sites offset from the
COM, so they produce torque and are implemented as a custom `ForceCompute`. The
nonbonded terms fit HOOMD's `AnisoPotentialPair` engine and are one evaluator each.

## Build

Requires a HOOMD-blue install (this repo is validated against 6.1.1 in the `main`
conda env) and cmake. From the repo root:

```bash
cmake -B build -S . -DHOOMD_DIR=$CONDA_PREFIX/lib/cmake/hoomd \
      -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
cmake --build build -j
cmake --install build          # installs _engine next to the oxdna/ package
```

## Use

```python
import hoomd, oxdna
from oxdna import io
from oxdna.model import dna2            # or dna1

snap = io.build_snapshot("system.top", "system.conf")
sim = hoomd.Simulation(device=hoomd.device.CPU())
sim.create_state_from_snapshot(snap)

forces, nlist = dna2.forces()           # all oxDNA2 terms, sequence-averaged
kT = dna2.dna1.kT_from_temperature(296.15)
lang = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
sim.operations.integrator = hoomd.md.Integrator(
    dt=0.003, integrate_rotational_dof=True, forces=forces, methods=[lang])
sim.run(10000)
```

Everything is in oxDNA simulation units (length 0.8518 nm, `kT = 0.1·T/300`).

## GPU

GPU support is built automatically when HOOMD was compiled with GPU support
(`ENABLE_HIP`). To run on the GPU, just change the device — nothing else:

```python
sim = hoomd.Simulation(device=hoomd.device.GPU())
```

The force wrappers dispatch to the `…GPU` classes automatically. All eight terms run
on the GPU: the five nonbonded terms via HOOMD's `AnisoPotentialPair` GPU kernels
(one evaluator each), and the orientation-aware bonded force (FENE + bonded excluded
volume + stacking) via a hand-written kernel that reuses the exact CPU physics
(`OxDNABondedKernel.h`) through HOOMD's per-particle bond table.

**Precision.** The `f1..f6` numerics (`OxDNANumerics.h`) are templated on the working
real type, and every force term computes in `ForceReal` (single precision on the
mixed-precision fork, double on vanilla HOOMD where `ForceReal` aliases `Scalar`) —
*except* the FENE backbone, which stays double. The nonbonded evaluators dominate
runtime and are dense in `acos`/`sin`/`exp`; HOOMD's aniso kernel already hands them a
`ForceReal` displacement, so single precision loses no accuracy the inputs did not
already carry, while avoiding the ~64× FP64 penalty on a consumer (gaming) GPU whose
transcendentals are otherwise emulated. The bonded kernel is mixed: its stacking and
bonded-excluded-volume math (the bulk of its cost) went single, but FENE stays double
because its `log(1 - dr²/δ²)` and `1/(δ²-dr²)` restoring force cancel catastrophically
in single precision near a stretched bond — the one term where double is a must.

On an RTX 4090, oxDNA2 on ~8k nucleotides runs at ~4600 steps/s (~5000 with the
optional fused path below) — ~7.8× the earlier all-double GPU build and ~940× the CPU
build; all per-term energies stay within the `< 1e-4` reference tolerance.
`--use_fast_math` was tried and measured ~20% *slower* — the hot transcendental is
`acos`, which has no fast intrinsic. The code stays compatible with both this fork and
upstream HOOMD via `MixedPrecisionCompat.h`.

**Fused hydrogen bonding + cross-stacking (`fused=True`).** These two terms act
between the same BASE sites and share their entire six-angle geometry. Running them as
separate `AnisoPotentialPair` forces means two GPU kernels each re-traverse the
neighbour list and re-rotate the quaternions; `EvaluatorPairOxDNAHBondCross` computes
that geometry once and applies both (reusing the *same* `base_base_term` code, no
duplicated math). `dna1.forces(fused=True)` / `dna2.forces(fused=True)` opt in — worth
~10% and numerically identical (validated in `pytest/test_fused.py`). It is off by
default because the fused force reports one combined energy rather than separate hb /
cr_stack terms; leave it off when you want per-term energies.

**Single-precision rotational integration (sloptimize fork).** Once the forces are
single precision, HOOMD's own rigid-body integrator becomes a large share of runtime;
the compute-bound part is the step-one no-squish rotation (~10 `cos`/`sin` per
nucleotide). The fork's `gpu_nve_angular_step_one_kernel` computes it in the
reduced-precision `ForceReal` type with `fast::` trig — exactly like the force
evaluators — so the precision follows the build (`SHORTREAL_SIZE`): single on the
mixed build, byte-identical double on the double/upstream builds (where `ForceReal`
aliases `Scalar` and `fast::` is the same `::cos`/`::sin`/`::sqrt` as `slow::`). No
runtime flag; nothing for the plugin to enable. The orientation/angmom state stays
double (`Scalar4`) and the translational position integration stays double regardless.
Worth ~6% on the mixed build, validated to give the same equilibrium energetics.

## Test

Via `ctest` (after building and installing — it uses HOOMD's interpreter and the
installed `_engine`):

```bash
cmake --build build && cmake --install build
ctest --test-dir build --output-on-failure
```

or directly with pytest:

```bash
PYTHONPATH=$PWD python -m pytest pytest/
```

Each energy term is validated per-nucleotide against oxDNA's `split_energy.dat`
reference to ~1e-6, with finite-difference force/torque checks and duplex-stability
runs. The reference data lives in the sibling `jax-dna/.../data/test-data` tree
(override with `OXDNA_TESTDATA`). Set `OXDNA_TEST_DEVICE=gpu` to run the whole suite
on the GPU.

## Status

- oxDNA1 (7 terms) and oxDNA2 (8 terms, grooved geometry + Debye-Hückel): complete
  and validated (sequence-averaged and sequence-dependent), on CPU and GPU.
- `bench/benchmark.py` times CPU vs GPU on a tiled system.
- Not yet implemented: oxRNA, sequence-dependent oxDNA2 tables, exact (COM) virial.
