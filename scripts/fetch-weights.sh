#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
work=${WORK_DIR:-"$(dirname "$scripts")/.work"}
model=${MODEL_DIR:-"$work/model"}
revision=7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6

if [[ ${1:-} == --help ]]; then
    echo 'Usage: fetch-weights.sh [--check]  (MODEL_DIR, WORK_DIR)'
    exit 0
fi
if [[ $# == 0 ]]; then
    mapfile -t files < <(awk '{print $2}' "$scripts/weights.sha256")
    hf download thinkingmachines/Inkling-2-Mini "${files[@]}" --revision "$revision" --local-dir "$model"
elif [[ $# != 1 || $1 != --check ]]; then
    echo 'Usage: fetch-weights.sh [--check]' >&2
    exit 2
fi
(cd -- "$model" && sha256sum --strict --check "$scripts/weights.sha256")
