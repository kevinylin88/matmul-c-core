#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/proj3_tune_matmul_bin}"
BENCH_C="$ROOT_DIR/tune_matmul/bench_matmul_params.c"
RESULTS_FILE="$(mktemp)"
mkdir -p "$BUILD_DIR"

CC="${CC:-gcc}"
DIM="${DIM:-1000}"
REPEAT="${REPEAT:-15}"
WARMUP="${WARMUP:-1}"
THREADS="${OMP_NUM_THREADS:-16}"
AUTO_REFINE="${AUTO_REFINE:-0}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

CFLAGS=(
  -O3
  -march=native
  -mavx2
  -mfma
  -mavx512f
  -fopenmp
  -I"$ROOT_DIR"
)

LDFLAGS=(
  -lopenblas
  -lm
)

# Broad 50-point sweep for DIM=1000.
DEFAULT_CANDIDATES=$(cat <<'PARAMS'
72 512 96
72 768 128
72 1024 160
72 1536 192
72 2048 224
84 512 128
84 768 160
84 1024 192
84 1536 224
84 2048 256
96 512 128
96 768 160
96 1024 192
96 1536 224
96 2048 256
96 3072 160
96 4096 192
108 768 128
108 1024 160
108 1536 192
108 2048 224
108 3072 256
120 768 128
120 1024 160
120 1536 192
120 2048 224
120 3072 256
120 4096 320
132 768 128
132 1024 160
132 1536 192
132 2048 224
132 3072 256
132 4096 320
132 8192 160
132 8192 224
132 8192 320
144 1024 128
144 1536 160
144 2048 192
144 3072 224
144 4096 256
144 8192 160
156 1024 128
156 1536 160
156 2048 192
156 3072 224
156 4096 256
156 8192 160
156 8192 320
PARAMS
)

CANDIDATES="${CANDIDATES:-$DEFAULT_CANDIDATES}"
BENCH_BIN="$BUILD_DIR/bench_matmul_params"

extract_field()
{
  local line="$1"
  local key="$2"
  awk -v key="$key" '{
    for(i = 1; i <= NF; i++){
      split($i, pair, "=");
      if(pair[1] == key){
        print pair[2];
        exit;
      }
    }
  }' <<< "$line"
}

already_tested()
{
  local ii="$1"
  local jj="$2"
  local kk="$3"
  grep -q "II=$ii JJ=$jj KK=$kk " "$RESULTS_FILE" 2>/dev/null
}

printf '==== compile parameterized bench ====\n'
"$CC" "${CFLAGS[@]}" \
  "$BENCH_C" \
  "$ROOT_DIR/multiply.c" \
  "$ROOT_DIR/kernel.c" \
  "${LDFLAGS[@]}" \
  -o "$BENCH_BIN"

run_one()
{
  local ii="$1"
  local jj="$2"
  local kk="$3"

  if already_tested "$ii" "$jj" "$kk"; then
    return
  fi

  printf '\n==== run II=%s JJ=%s KK=%s DIM=%s REPEAT=%s THREADS=%s ====\n' \
    "$ii" "$jj" "$kk" "$DIM" "$REPEAT" "$THREADS"

  DIM="$DIM" REPEAT="$REPEAT" WARMUP="$WARMUP" "$BENCH_BIN" "$ii" "$jj" "$kk" | tee /tmp/tune_matmul_current.txt
  grep '^RESULT ' /tmp/tune_matmul_current.txt >> "$RESULTS_FILE"
}

print_fastest()
{
  local count="${1:-10}"
  printf '\n==== fastest by MED_MS ====\n'
  awk '{
    med = "";
    for(i = 1; i <= NF; i++){
      split($i, pair, "=");
      if(pair[1] == "MED_MS"){
        med = pair[2];
      }
    }
    if(med != ""){
      print med, $0;
    }
  }' "$RESULTS_FILE" | sort -n | head -n "$count" | cut -d' ' -f2-
}

while read -r ii jj kk; do
  if [[ -z "${ii:-}" || "${ii:0:1}" == "#" ]]; then
    continue
  fi
  run_one "$ii" "$jj" "$kk"
done <<< "$CANDIDATES"

print_fastest 5

if [[ "$AUTO_REFINE" == "1" ]]; then
  best_line=$(awk '{
    med = "";
    for(i = 1; i <= NF; i++){
      split($i, pair, "=");
      if(pair[1] == "MED_MS"){
        med = pair[2];
      }
    }
    if(med != ""){
      print med, $0;
    }
  }' "$RESULTS_FILE" | sort -n | head -n 1 | cut -d' ' -f2-)

  best_ii=$(extract_field "$best_line" "II")
  best_jj=$(extract_field "$best_line" "JJ")
  best_kk=$(extract_field "$best_line" "KK")

  printf '\n==== refine around II=%s JJ=%s KK=%s ====\n' "$best_ii" "$best_jj" "$best_kk"

  for params in \
    "$((best_ii - 12)) $best_jj $best_kk" \
    "$((best_ii + 12)) $best_jj $best_kk" \
    "$best_ii $((best_jj - 128)) $best_kk" \
    "$best_ii $((best_jj + 128)) $best_kk" \
    "$best_ii $best_jj $((best_kk - 32))" \
    "$best_ii $best_jj $((best_kk + 32))" \
    "$((best_ii - 12)) $((best_jj - 128)) $best_kk" \
    "$((best_ii + 12)) $((best_jj + 128)) $best_kk"
  do
    read -r ii jj kk <<< "$params"
    if (( ii >= 72 && jj >= 256 && kk >= 96 )); then
      run_one "$ii" "$jj" "$kk"
    fi
  done

  print_fastest 5
fi
