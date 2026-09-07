#!/usr/bin/env bash
# Check that public discovery surfaces are filled after launch.
# Usage: assert-launch.sh OWNER/REPO
# Exit 0 launched, 1 missing surface, 2 cannot query.
set -euo pipefail

echo "PLAN: assert launch metadata for ${1:-missing}"

if [[ $# -ne 1 || "$1" != */* ]]; then
  echo "FAIL: usage: assert-launch.sh OWNER/REPO"
  echo "DONE: ok=false error=usage"
  exit 2
fi

repo="$1"
missing=0

if ! command -v gh >/dev/null 2>&1; then
  echo "FAIL: gh not on PATH"
  echo "DONE: ok=false error=no-gh"
  exit 2
fi

echo "DO: query GitHub repo About"
if ! repo_json="$(gh repo view "$repo" --json description,repositoryTopics,isPrivate 2>/dev/null)"; then
  echo "FAIL: gh repo view $repo"
  echo "DONE: ok=false error=gh-repo-view"
  exit 2
fi

desc="$(printf '%s' "$repo_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("description") or "")')"
topics="$(printf '%s' "$repo_json" | python3 -c 'import json,sys; t=json.load(sys.stdin).get("repositoryTopics") or []; print(",".join(x.get("name","") if isinstance(x,dict) else str(x) for x in t))')"
private="$(printf '%s' "$repo_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("isPrivate"))')"

if [[ -z "$desc" || "$desc" == "null" ]]; then
  echo "FAIL: description empty"
  missing=$((missing + 1))
else
  echo "OK: description set"
fi

if [[ -z "$topics" ]]; then
  echo "FAIL: topics empty"
  missing=$((missing + 1))
else
  echo "OK: topics set"
fi

if [[ "$private" == "True" || "$private" == "true" ]]; then
  echo "FAIL: repo is private"
  missing=$((missing + 1))
else
  echo "OK: repo is public"
fi

remote_is_repo() {
  local dir="$1"
  local want="$2"
  local remotes
  remotes="$(git -C "$dir" remote -v 2>/dev/null || true)"
  if [[ -z "$remotes" ]]; then
    return 1
  fi
  if printf '%s\n' "$remotes" | grep -Eqi "github\\.com[:/]${want}(\\.git)?[[:space:]]"; then
    return 0
  fi
  return 1
}

echo "DO: resolve local checkout of $repo"
root=""
script_dir="$(cd "$(dirname "$0")" && pwd)"
script_top="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$script_top" ]] && remote_is_repo "$script_top" "$repo"; then
  root="$script_top"
  echo "OK: using script tree $root"
else
  cwd_top="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$cwd_top" ]] && remote_is_repo "$cwd_top" "$repo"; then
    root="$cwd_top"
    echo "OK: using cwd tree $root"
  else
    echo "OK: no local checkout of $repo; skip file scan"
  fi
fi

if [[ -n "$root" ]]; then
  if [[ ! -f "$root/README.md" ]]; then
    echo "FAIL: README.md missing"
    missing=$((missing + 1))
  elif grep -qx 'Not ready.' "$root/README.md"; then
    echo "FAIL: README is still the stealth one-liner"
    missing=$((missing + 1))
  else
    echo "OK: README is a launch page"
  fi

  if [[ ! -f "$root/llms.txt" ]]; then
    echo "FAIL: llms.txt missing"
    missing=$((missing + 1))
  else
    echo "OK: llms.txt present"
  fi

  if [[ -f "$root/Cargo.toml" ]]; then
    desc_crate="$(ROOT="$root" python3 - <<'PY'
import os, pathlib, re
t = (pathlib.Path(os.environ["ROOT"]) / "Cargo.toml").read_text()
m = re.search(r'(?m)^description\s*=\s*"(.*)"', t)
print(m.group(1) if m else "")
PY
)"
    kw="$(ROOT="$root" python3 - <<'PY'
import os, pathlib, re
t = (pathlib.Path(os.environ["ROOT"]) / "Cargo.toml").read_text()
m = re.search(r"(?m)^keywords\s*=\s*\[([^\]]*)\]", t)
print((m.group(1) if m else "").strip())
PY
)"
    if [[ -z "$desc_crate" || "$desc_crate" == "Reserved." ]]; then
      echo "FAIL: Cargo.toml description is a stub"
      missing=$((missing + 1))
    else
      echo "OK: Cargo.toml description set"
    fi
    if [[ -z "$kw" ]]; then
      echo "FAIL: Cargo.toml keywords empty"
      missing=$((missing + 1))
    else
      echo "OK: Cargo.toml keywords set"
    fi
  fi
fi

if [[ "$missing" -gt 0 ]]; then
  echo "DONE: ok=false missing=$missing"
  echo "NEXT: fill About, topics, README, and crate identity"
  exit 1
fi

echo "DONE: ok=true missing=0"
echo "NEXT: crates.io stays a separate human publish"
exit 0
