#!/usr/bin/env bash
# Replay a cached probe as host-policy JSON. Does not seed the cache.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CACHE MODEL [PROVIDER]" >&2
  exit 2
fi

cache=$1
model=$2
provider=${3:-ollama}

if [[ ! -f $cache ]]; then
  echo "missing cache file: $cache" >&2
  exit 2
fi

canact probe --json --model "$model" --provider "$provider" --cache "$cache"
