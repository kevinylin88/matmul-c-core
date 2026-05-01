#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/correctness/bin"
mkdir -p "$OUT_DIR"

THREADS="${OMP_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

BIN="$OUT_DIR/test_v8_correctness"

gcc \
  -O3 \
  -march=native \
  -mavx2 \
  -mfma \
  -mavx512f \
  -fopenmp \
  -I"$ROOT_DIR" \
  "$ROOT_DIR/correctness/test_matmul_correctness.c" \
  "$ROOT_DIR/multiply.c" \
  "$ROOT_DIR/kernel.c" \
  -lopenblas \
  -lm \
  -o "$BIN"

"$BIN"
