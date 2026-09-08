#!/usr/bin/env bash
# Publish canact to crates.io when this tree's version is not there yet.
# Skip (exit 0) if crates.io already has the version or cargo says
# already uploaded. Missing source or a real publish error is exit 1.
set -euo pipefail

CRATE="${CRATE:-canact}"
TAG="${TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
CARGO="${CARGO:-cargo}"
CURL="${CURL:-curl}"
USER_AGENT="${USER_AGENT:-canact-publish (https://github.com/canact/canact)}"

echo "PLAN: publish ${CRATE} from $(pwd)"

if [ ! -f Cargo.toml ]; then
  echo "FAIL: Cargo.toml not in $(pwd)" >&2
  exit 1
fi

VERSION="$(python3 - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path("Cargo.toml").read_text(encoding="utf-8"))["package"]["version"])
PY
)"

if [ -z "${VERSION}" ]; then
  echo "FAIL: no package.version in Cargo.toml" >&2
  exit 1
fi

if [ -n "${TAG}" ]; then
  semver="${TAG#v}"
  if [ "${semver}" != "${VERSION}" ]; then
    echo "FAIL: tag ${TAG} does not match Cargo.toml ${VERSION}" >&2
    exit 1
  fi
fi

echo "PLAN: ${CRATE} ${VERSION}"

tmp="$(mktemp)"
cleanup() { rm -f "${tmp}"; }
trap cleanup EXIT

http="$("${CURL}" -sS -o "${tmp}" -w '%{http_code}' -A "${USER_AGENT}" \
  "https://crates.io/api/v1/crates/${CRATE}/${VERSION}" || true)"

if [ "${http}" = "200" ]; then
  echo "OK: ${CRATE} ${VERSION} already on crates.io"
  exit 0
fi

if [ "${http}" != "404" ]; then
  echo "FAIL: crates.io GET ${CRATE}/${VERSION} HTTP ${http}" >&2
  if [ -s "${tmp}" ]; then
    head -c 400 "${tmp}" >&2
    echo >&2
  fi
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN: would cargo publish --locked ${CRATE} ${VERSION}"
  exit 0
fi

if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "FAIL: CARGO_REGISTRY_TOKEN is unset" >&2
  exit 1
fi

echo "DO: ${CARGO} publish --locked"
set +e
"${CARGO}" publish --locked >"${tmp}" 2>&1
st=$?
set -e
cat "${tmp}"

if [ "${st}" -eq 0 ]; then
  echo "DONE: published ${CRATE} ${VERSION}"
  exit 0
fi

if grep -qiE 'already uploaded|already exists' "${tmp}"; then
  echo "OK: cargo reported already uploaded"
  exit 0
fi

echo "FAIL: cargo publish exited ${st}" >&2
exit "${st}"
