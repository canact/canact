#!/usr/bin/env bash
# cargo deny plus a hard fail if a bline crate appears in Cargo.toml.
set -euo pipefail
echo "PLAN: deny bline-* deps and run cargo deny"
if grep -E '^\s*bline-(llm|types|probe|core|cli)\s*=' Cargo.toml; then
  echo "FAIL: bline crate in Cargo.toml"
  exit 1
fi
echo "OK: no bline crates in Cargo.toml"
cargo deny check
echo "DONE"
