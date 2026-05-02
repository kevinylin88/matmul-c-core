# Project 3 Matrix Multiplication

This project implements and benchmarks several SGEMM-style matrix
multiplication kernels in C. The current optimized path is
`matmul_v8_avx512_omp_improved`, which uses OpenMP, AVX512 micro-kernels,
packing, and a panel-B layout for the main 32-column AVX512 path.

## Files

- `main.c`: simple performance benchmark driver.
- `multiply.c`: matrix multiplication implementations from plain C to v8 and
  OpenBLAS wrapper.
- `kernel.c`: AVX2 and AVX512 micro-kernels.
- `multiply.h`: shared matrix type, constants, and function declarations.
- `correctness/`: correctness tests for v8 against plain C and OpenBLAS.
- `tune_matmul/`: parameter tuning benchmark for `II`, `JJ`, and `KK`.

## Current V8 Parameters

The default v8 compile-time parameters are in `multiply.c`:

```c
#ifndef II
#define II 132
#endif
#ifndef JJ
#define JJ 8640
#endif
#ifndef KK
#define KK 160
#endif
```

The B panel width is defined in `multiply.h`:

```c
#define NR 32
```

V8 packs B as 32-column panels:

```c
B_pack[panel][kd][inner]
panel = jd / NR
inner = jd % NR
```

The AVX512 main path uses the `*_panelb` kernels in `kernel.c`. Tail columns
that are not a multiple of 32 are handled by scalar panel-aware fallback code
inside v8.

## Build

Normal build:

```bash
gcc -O3 -g -march=native -mavx2 -mfma -mavx512f -fopenmp \
  main.c multiply.c kernel.c -lopenblas -lm -o main
```

Build with LTO:

```bash
gcc -O3 -flto -march=native -mavx2 -mfma -mavx512f -fopenmp \
  main.c multiply.c kernel.c -lopenblas -lm -o main
```

Run:

```bash
./main
```

The optimized v8 path requires AVX512 support. Running it on a CPU without
AVX512 support may fail with an illegal instruction.

## Correctness Test

Run the v8 correctness suite:

```bash
env OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  bash correctness/test_matmul_correctness.sh
```

Expected final line:

```text
all v8 correctness tests passed
```

The correctness test covers small irregular sizes and large square/irregular
sizes. Small cases compare against plain C; large cases compare against
OpenBLAS.

## Tuning

Run the current tuning script:

```bash
bash tune_matmul/run_tune_matmul.sh | tee tune_matmul/tunedata_4000_panelb.txt
```

Current script defaults:

```text
DIM=4000
REPEAT=15
WARMUP=1
THREADS=16 unless OMP_NUM_THREADS is already set
```

Override defaults from the command line:

```bash
DIM=8000 REPEAT=15 bash tune_matmul/run_tune_matmul.sh | tee tune_matmul/tunedata_8000_panelb.txt
```

The tuning benchmark calls:

```c
matmul_v8_avx512_omp_improved(mat1, mat2, mat3, ii, jj, kk);
```

The script prints `AVG_MS`, `MED_MS`, `BEST_MS`, `WORST_MS`, `AVG_GFLOPS`,
`BEST_GFLOPS`, and `MED_GFLOPS`. Prefer `MED_GFLOPS` or `MED_MS` when results
contain outliers.

## Benchmarking Notes

For fair comparison between v8 and OpenBLAS:

- Keep matrix allocation and initialization outside the timed region.
- Clear C before each timed multiply.
- Measure each implementation in full blocks, not alternating every iteration.
- Use median time as the primary result.
- Set thread counts explicitly.

Example environment:

```bash
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

For single-thread comparisons:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## OpenBLAS Wrapper

`matmul_v9_OpenBLAS` calls `cblas_sgemm` in row-major mode:

```c
C = A * B
```

with `beta = 0.0f`, so OpenBLAS overwrites C. Most custom kernels accumulate
with `+=`, so C must be zeroed before calling them.
