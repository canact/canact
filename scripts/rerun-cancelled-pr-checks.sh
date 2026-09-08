#!/usr/bin/env bash
# Rerun a cancelled Actions run when its SHA is still an open PR head.
# A double synchronize can cancel the current SHA and never start a
# replacement (canact #162 Semantic PR Title).
#
# Env:
#   GH_REPO      owner/name
#   HEAD_SHA     check_run.head_sha
#   RUN_ID       workflow run id (optional if DETAILS_URL is set)
#   DETAILS_URL  check_run.details_url (parsed for /actions/runs/ID)
#   MAX_ATTEMPT  skip rerun at or above this attempt (default 2)
#   DRY_RUN=1    print the decision, do not rerun
set -euo pipefail

REPO="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
HEAD_SHA="${HEAD_SHA:-}"
RUN_ID="${RUN_ID:-}"
DETAILS_URL="${DETAILS_URL:-}"
MAX_ATTEMPT="${MAX_ATTEMPT:-2}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "${REPO}" ]; then
  echo "FAIL: GH_REPO or GITHUB_REPOSITORY required" >&2
  exit 1
fi

if [ -z "${HEAD_SHA}" ]; then
  echo "FAIL: HEAD_SHA required" >&2
  exit 1
fi

if [ -z "${RUN_ID}" ] && [[ "${DETAILS_URL}" =~ actions/runs/([0-9]+) ]]; then
  RUN_ID="${BASH_REMATCH[1]}"
fi

if [ -z "${RUN_ID}" ]; then
  echo "OK: no RUN_ID (workflow_dispatch); nothing to rerun"
  exit 0
fi

echo "PLAN: run ${RUN_ID} sha ${HEAD_SHA} repo ${REPO}"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "FAIL: GH_TOKEN required" >&2
  exit 1
fi

heads="$(gh pr list --repo "${REPO}" --state open --json headRefOid --jq '.[].headRefOid')"
current=0
while IFS= read -r head; do
  if [ "${head}" = "${HEAD_SHA}" ]; then
    current=1
    break
  fi
done <<<"${heads}"

if [ "${current}" != "1" ]; then
  echo "OK: ${HEAD_SHA} is not an open PR head; leave run ${RUN_ID}"
  exit 0
fi

meta="$(gh run view "${RUN_ID}" --repo "${REPO}" --json conclusion,attempt,headSha,status)"
conclusion="$(printf '%s\n' "${meta}" | jq -r '.conclusion')"
attempt="$(printf '%s\n' "${meta}" | jq -r '.attempt')"
run_sha="$(printf '%s\n' "${meta}" | jq -r '.headSha')"

if [ "${run_sha}" != "${HEAD_SHA}" ]; then
  echo "OK: run ${RUN_ID} sha ${run_sha} != ${HEAD_SHA}; skip"
  exit 0
fi

if [ "${conclusion}" != "cancelled" ]; then
  echo "OK: run ${RUN_ID} conclusion ${conclusion}; skip"
  exit 0
fi

if [ "${attempt}" -ge "${MAX_ATTEMPT}" ]; then
  echo "OK: run ${RUN_ID} attempt ${attempt} >= ${MAX_ATTEMPT}; skip"
  exit 0
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN: would gh run rerun ${RUN_ID}"
  exit 0
fi

echo "DO: gh run rerun ${RUN_ID}"
gh run rerun "${RUN_ID}" --repo "${REPO}"
echo "DONE: reran ${RUN_ID}"
