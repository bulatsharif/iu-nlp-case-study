#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config_name>"
    exit 2
fi

python runner.py --config "$1"
