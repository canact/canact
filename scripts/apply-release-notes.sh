#!/usr/bin/env bash
# Apply docs/releases/<tag>.md to an existing GitHub Release.
# Missing file is a no-op (auto changelog stays). The file is never
# deleted from git.
set -euo pipefail

TAG="${TAG:-}"
REPO="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
DRY_RUN="${DRY_RUN:-0}"
NOTES_PATH="docs/releases/${TAG}.md"

if [[ ! "${TAG}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "TAG must look like vX.Y.Z: ${TAG}" >&2
  exit 1
fi

if [ -z "${REPO}" ]; then
  echo "GH_REPO or GITHUB_REPOSITORY required" >&2
  exit 1
fi

tmp="$(mktemp)"
cleanup() { rm -f "${tmp}"; }
trap cleanup EXIT

if [ -f "${NOTES_PATH}" ]; then
  echo "PLAN: local ${NOTES_PATH}"
  cp "${NOTES_PATH}" "${tmp}"
elif [ -n "${GH_TOKEN:-}" ]; then
  echo "PLAN: fetch ${NOTES_PATH} from main"
  if ! gh api "repos/${REPO}/contents/${NOTES_PATH}?ref=main" \
    -H "Accept: application/vnd.github.raw" >"${tmp}"; then
    echo "OK: no ${NOTES_PATH} on this tree or main; leaving auto notes"
    exit 0
  fi
else
  echo "OK: no ${NOTES_PATH} and no GH_TOKEN; leaving auto notes"
  exit 0
fi

if [ ! -s "${tmp}" ]; then
  echo "OK: ${NOTES_PATH} is empty; leaving auto notes"
  exit 0
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN: would apply ${NOTES_PATH} to ${TAG}"
  echo "BYTES: $(wc -c <"${tmp}" | tr -d ' ')"
  exit 0
fi

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "FAIL: release ${TAG} does not exist" >&2
  exit 1
fi

echo "DO: gh release edit ${TAG}"
gh release edit "${TAG}" --repo "${REPO}" --notes-file "${tmp}"
echo "DONE: applied ${NOTES_PATH} to ${TAG}"
