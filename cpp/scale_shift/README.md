# Streaming least-squares PRO scale/shift (C1b)

`calibrated = α · raw + β`. Same equations as
`python -m spur_depth.calib.fit_scale_shift`.

The training loaders used α = `-0.06610793956568871`, β = `1.555980697834118`.
Those constants now live in `spur_depth/calib/pro_best_config.json`. This
binary re-derives them from `.npy` pairs. **`Data/full_spur` is not mounted**,
so the 24k-frame wall-time table below is empty on purpose — filling it in
with a laptop-scale synthetic run and calling it 24k would be a lie.

## Build

```bash
# CI / machines with cmake
cmake --preset ci
cmake --build --preset ci
ctest --preset ci

# This login node (g++ 12, no cmake on PATH)
make -C cpp/scale_shift test
```

## Usage

```bash
# CSV: pred.npy,gt.npy,mask.npy  (sorted merge is by CSV order here;
# Python sorts by pred path — keep the CSV sorted if you compare outputs)
./scale_shift_fit --pairs-csv pairs.csv --threads 8 --erode-r 10
```

OpenMP parallelises **files**. Each file's moments are computed on one
thread; the 2×2 solve merges those structs in file-index order. That is
why 1-thread and 8-thread results are bit-identical (see the smoke test).
An `omp reduction` on the floats themselves would reorder the sum and
break that claim.

## Benchmark (24k `.npy`, ~88 GB)

I/O-bound. Report cold and warm cache separately. Disk, not arithmetic,
is the wall — do not expect 8× from 8 threads.

| Implementation | Wall (cold) | Wall (warm) | α | β | max abs vs Python |
| --- | --- | --- | --- | --- | --- |
| Python + NumPy | — | — | not re-scored | not re-scored | — |
| C++17, 1 thread | — | — | | | |
| C++17, 8 threads (OpenMP) | — | — | | | |

Fill this table after `Data/full_spur` is restored. Until then the gate is
the synthetic ramp (α/β recovered to 1e-6 from float32 `.npy`; Python float64 hits 1e-9) and the 1-vs-8-thread bit-identity
test, both of which run in CI.
