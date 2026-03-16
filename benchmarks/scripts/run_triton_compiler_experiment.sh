#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/scripts/run_triton_compiler_experiment.sh \
    --upstream-triton /path/to/upstream-triton \
    --hack-triton /path/to/triton-sw-hack \
    [--python python] \
    [--run-opv9]

This wrapper installs one compiler at a time into the current Python env,
benchmarks OPv6 under each compiler, then compares the saved outputs.

If --run-opv9 is provided, it also benchmarks OPv9 after installing hack-v2.
EOF
}

PYTHON_BIN="python"
UPSTREAM_TRITON=""
HACK_TRITON=""
RUN_OPV9="0"
EXPERIMENT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --upstream-triton)
      UPSTREAM_TRITON="$2"
      shift 2
      ;;
    --hack-triton)
      HACK_TRITON="$2"
      shift 2
      ;;
    --run-opv9)
      RUN_OPV9="1"
      shift
      ;;
    --)
      shift
      EXPERIMENT_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${UPSTREAM_TRITON}" || -z "${HACK_TRITON}" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${REPO_ROOT}/benchmarks/results/compiler_compare/${STAMP}"
COMPARE_OUT="${OUTPUT_ROOT}/comparison"

install_compiler() {
  local compiler_path="$1"
  echo "==> Installing Triton from ${compiler_path}"
  "${PYTHON_BIN}" -m pip uninstall -y triton >/dev/null 2>&1 || true
  TRITON_BUILD_TESTING=OFF "${PYTHON_BIN}" -m pip install -e "${compiler_path}"
}

run_experiment() {
  local label="$1"
  local compiler_root="$2"
  local kernel="$3"

  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/scripts/experiment_triton_compiler.py" \
    --compiler-label "${label}" \
    --compiler-root "${compiler_root}" \
    --kernel "${kernel}" \
    --output-root "${OUTPUT_ROOT}" \
    "${EXPERIMENT_ARGS[@]}"
}

install_compiler "${UPSTREAM_TRITON}"
run_experiment "upstream" "${UPSTREAM_TRITON}" "opv6"
UPSTREAM_SUMMARY="$(find "${OUTPUT_ROOT}" -maxdepth 2 -name 'summary.json' | sort | grep 'upstream_opv6' | tail -n 1)"

install_compiler "${HACK_TRITON}"
run_experiment "hack-v2" "${HACK_TRITON}" "opv6"
HACK_OPV6_SUMMARY="$(find "${OUTPUT_ROOT}" -maxdepth 2 -name 'summary.json' | sort | grep 'hack-v2_opv6' | tail -n 1)"

"${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/scripts/compare_triton_compiler_runs.py" \
  --run-a "${UPSTREAM_SUMMARY}" \
  --run-b "${HACK_OPV6_SUMMARY}" \
  --output-dir "${COMPARE_OUT}"

if [[ "${RUN_OPV9}" == "1" ]]; then
  run_experiment "hack-v2" "${HACK_TRITON}" "opv9"
fi

echo "Results saved under ${OUTPUT_ROOT}"
