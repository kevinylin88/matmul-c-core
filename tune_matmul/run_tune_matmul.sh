#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/proj3_tune_matmul_bin}"
BENCH_C="$ROOT_DIR/tune_matmul/bench_matmul_params.c"
RESULTS_FILE="$(mktemp)"
mkdir -p "$BUILD_DIR"

CC="${CC:-gcc}"
DIM="${DIM:-4000}"
REPEAT="${REPEAT:-15}"
WARMUP="${WARMUP:-1}"
THREADS="${OMP_NUM_THREADS:-16}"
AUTO_REFINE="${AUTO_REFINE:-1}"

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

# Candidates are centered on the best DIM=8000 panel-B results:
#   168 832 224, 168 704 224, 168 704 256, 168 768 224, 180 832 224.
# DIM=4000 has a different edge/tile interaction, so keep a few smaller JJ/KK
# baselines and sweep around the winning larger-JJ panel-B region.
DEFAULT_CANDIDATES=$(cat <<'PARAMS'
132 512 160
132 640 160
144 640 160
168 640 160
156 576 224
156 704 224
168 576 224
168 640 224
168 704 224
168 704 192
168 704 256
168 768 224
168 832 224
168 832 192
168 832 256
168 896 224
180 704 224
180 768 224
180 832 224
180 832 256
PARAMS
)

CANDIDATES="${CANDIDATES:-$DEFAULT_CANDIDATES}"

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

run_one()
{
  local ii="$1"
  local jj="$2"
  local kk="$3"

  if already_tested "$ii" "$jj" "$kk"; then
    return
  fi

  local bin="$BUILD_DIR/bench_matmul_ii${ii}_jj${jj}_kk${kk}"
  printf '\n==== compile/run II=%s JJ=%s KK=%s DIM=%s REPEAT=%s THREADS=%s ====\n' \
    "$ii" "$jj" "$kk" "$DIM" "$REPEAT" "$THREADS"

  "$CC" "${CFLAGS[@]}" \
    -DII="$ii" -DJJ="$jj" -DKK="$kk" \
    "$BENCH_C" \
    "$ROOT_DIR/multiply.c" \
    "$ROOT_DIR/kernel.c" \
    "${LDFLAGS[@]}" \
    -o "$bin"

  DIM="$DIM" REPEAT="$REPEAT" WARMUP="$WARMUP" "$bin" | tee /tmp/tune_matmul_current.txt
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
