#!/usr/bin/env bash
# cargo deny plus a hard fail if a bline-* crate appears in Cargo.toml or Cargo.lock.
set -euo pipefail
echo "PLAN: deny bline-* deps and run cargo deny"
if grep -E '^\s*bline-[A-Za-z0-9_-]+\s*=' Cargo.toml; then
  echo "FAIL: bline crate in Cargo.toml"
  exit 1
fi
echo "OK: no bline crates in Cargo.toml"
if grep -E '^name = "bline-' Cargo.lock; then
  echo "FAIL: bline crate in Cargo.lock"
  exit 1
fi
echo "OK: no bline crates in Cargo.lock"
echo "DO: cargo deny check"
cargo deny check
echo "DONE"
