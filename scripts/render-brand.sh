#!/usr/bin/env bash
# Rasterize docs/brand/canact.svg. Derived PNG stays out of git.
set -euo pipefail

echo "PLAN: rasterize docs/brand/canact.svg"

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/docs/brand/canact.svg"
out="${CANACT_BRAND_OUT:-/tmp/canact-brand}"
rsvg="${RSVG_CONVERT:-/opt/homebrew/bin/rsvg-convert}"

if [[ ! -f "$src" ]]; then
  echo "FAIL: missing $src"
  echo "DONE: ok=false"
  exit 1
fi
if [[ ! -x "$rsvg" ]]; then
  if command -v rsvg-convert >/dev/null; then
    rsvg="$(command -v rsvg-convert)"
  else
    echo "FAIL: rsvg-convert not found"
    echo "DONE: ok=false"
    exit 1
  fi
fi
if ! command -v magick >/dev/null; then
  echo "FAIL: magick not found"
  echo "DONE: ok=false"
  exit 1
fi

echo "DO: mkdir $out"
mkdir -p "$out"

echo "DO: org avatar 1024"
"$rsvg" -w 1024 -h 1024 "$src" -o "$out/org-avatar-1024.png"

echo "DO: social preview 1280x640"
"$rsvg" -w 480 -h 480 "$src" -o "$out/mark-480.png"
magick -size 1280x640 xc:'#0B1220' \
  \( "$out/mark-480.png" \) \
  -gravity center -compose over -composite \
  PNG24:"$out/social-preview.png"
rm -f "$out/mark-480.png"

echo "OK: $out/org-avatar-1024.png"
echo "OK: $out/social-preview.png"
echo "DONE: ok=true"
echo "NEXT: upload PNG from $out; keep $src as the source"
