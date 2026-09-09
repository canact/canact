#!/usr/bin/env bash
# Sign an existing release tag in place. Does not move the commit.
# Used by .github/workflows/sign-tags.yml and for local backfill.
set -euo pipefail

echo "PLAN: GPG-sign git tag ${TAG:-<empty>} at its current SHA"

if [[ ! "${TAG:-}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "FAIL: TAG must look like vX.Y.Z: ${TAG:-<empty>}" >&2
  exit 1
fi

if ! git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "FAIL: tag ${TAG} is not in this clone" >&2
  exit 1
fi

if [[ "$(git cat-file -t "refs/tags/${TAG}")" == "tag" ]] \
  && git verify-tag "${TAG}" >/dev/null 2>&1; then
  echo "OK: ${TAG} is already a signed annotated tag"
  echo "DONE: no push"
  exit 0
fi

sha="$(git rev-list -n 1 "refs/tags/${TAG}")"
msg="canact ${TAG#v}"

echo "DO: git tag -s -f ${TAG} ${sha}"
git tag -s -f -m "${msg}" "${TAG}" "${sha}"
git verify-tag "${TAG}"

if [[ -z "${GH_TOKEN:-}" || -z "${GH_REPO:-}" ]]; then
  echo "OK: signed ${TAG} locally; GH_TOKEN/GH_REPO unset so no push"
  echo "DONE: ${TAG} -> ${sha}"
  exit 0
fi

echo "DO: force-push refs/tags/${TAG} to ${GH_REPO}"
git push "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" \
  "+refs/tags/${TAG}"
echo "DONE: ${TAG} -> ${sha}"
