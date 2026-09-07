#!/usr/bin/env bash
# Retired after launch. Public surfaces are checked by assert-launch.sh.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/assert-launch.sh" "$@"
