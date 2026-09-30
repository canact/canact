#!/usr/bin/env bash
# Generate a CycloneDX JSON SBOM and upload canact-sbom.cdx.json.
#
# cargo-cyclonedx --override-filename writes canact-sbom.json (no .cdx).
# Copy the single canact-sbom*.json to canact-sbom.cdx.json.
#
# Optional environment:
#   CYCLONEDX_VERSION  default 0.5.9
#   OUT                default canact-sbom.cdx.json
#   SEARCH_DIR         default .
#   DRY_RUN=1          print the install and generate commands and exit
#   FIXTURE            copy this file to OUT and skip generate
#   COLLECT_ONLY=1     find an existing canact-sbom*.json and copy it
#   SKIP_INSTALL=1     cargo-cyclonedx is already on PATH
#   SKIP_UPLOAD=1      do not call gh release upload
#   TAG, REPO, GH_TOKEN  required for a real upload
# TAG, when set, must look like vX.Y.Z (no stealth suffix).
set -euo pipefail

version="${CYCLONEDX_VERSION:-0.5.9}"
out="${OUT:-canact-sbom.cdx.json}"
search="${SEARCH_DIR:-.}"

echo "PLAN: stage CycloneDX SBOM ${out}"

if [ -n "${TAG:-}" ]; then
  if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "FAIL: TAG must look like vX.Y.Z: ${TAG}" >&2
    exit 1
  fi
fi

if [ "${DRY_RUN:-}" = "1" ] && [ -z "${FIXTURE:-}" ] && [ "${COLLECT_ONLY:-}" != "1" ]; then
  echo "DO: cargo install cargo-cyclonedx --version ${version} --locked"
  echo "DO: cargo cyclonedx --format json --features cli --override-filename canact-sbom"
  echo "OK: dry-run ${out}"
  echo "DONE: dry-run"
  exit 0
fi

collect_sbom() {
  local matches=()
  local path
  local src
  while IFS= read -r -d '' path; do
    matches+=("$path")
  done < <(find "$search" -maxdepth 2 -type f -name 'canact-sbom*.json' -print0)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "FAIL: expected one canact-sbom*.json under ${search}, found ${#matches[@]}" >&2
    exit 1
  fi
  src="${matches[0]}"
  if [ "$src" -ef "$out" ]; then
    echo "OK: ${out} already holds the SBOM"
    return 0
  fi
  echo "DO: copy $(basename "$src") to ${out}"
  cp "$src" "$out"
  echo "OK: wrote ${out}"
}

if [ -n "${FIXTURE:-}" ]; then
  if [ ! -f "$FIXTURE" ]; then
    echo "FAIL: FIXTURE is not a file: ${FIXTURE}" >&2
    exit 1
  fi
  echo "DO: copy fixture to ${out}"
  cp "$FIXTURE" "$out"
  echo "OK: wrote ${out}"
elif [ "${COLLECT_ONLY:-}" = "1" ]; then
  collect_sbom
else
  if [ "${SKIP_INSTALL:-}" != "1" ]; then
    echo "DO: cargo install cargo-cyclonedx --version ${version} --locked"
    cargo install cargo-cyclonedx --version "$version" --locked
  else
    echo "OK: skip cargo install"
  fi
  echo "DO: cargo cyclonedx --format json --features cli --override-filename canact-sbom"
  # A previous copy leaves canact-sbom.cdx.json beside the new canact-sbom.json.
  find "$search" -maxdepth 2 -type f -name 'canact-sbom*.json' -delete
  cargo cyclonedx --format json --features cli --override-filename canact-sbom
  collect_sbom
fi

if [ "${SKIP_UPLOAD:-}" = "1" ] || [ "${DRY_RUN:-}" = "1" ]; then
  echo "OK: skip upload"
  echo "DONE: staged ${out}"
  exit 0
fi

if [ -z "${TAG:-}" ] || [ -z "${REPO:-}" ] || [ -z "${GH_TOKEN:-}" ]; then
  echo "FAIL: TAG, REPO, and GH_TOKEN are required to upload" >&2
  exit 1
fi

echo "DO: gh release upload ${TAG} ${out}"
gh release upload "$TAG" "$out" --repo "$REPO" --clobber
echo "OK: uploaded ${out}"
echo "DONE: staged ${out}"
