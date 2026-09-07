#!/usr/bin/env python3
"""Generate or update the Scoop bucket manifest for canact.

Release CI runs this after GitHub Release assets exist so
canact/scoop-bucket pins the new version and SHA256. Client-side
Scoop checkver/autoupdate is a fallback only. The committed JSON is
the source of truth.

canact ships one Windows archive today: x86_64-pc-windows-msvc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAG_PREFIX = "v"
X64_ZIP = "canact-x86_64-pc-windows-msvc.zip"
HASH_RE = re.compile(r"^([a-fA-F0-9]{64})\b")


def parse_hash_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"empty hash file: {path}")
    m = HASH_RE.match(text.splitlines()[0].strip())
    if not m:
        raise SystemExit(f"could not parse SHA256 from {path}: {text!r}")
    return m.group(1).lower()


def find_hash(artifacts_dir: Path, zip_name: str) -> str:
    """Locate zip.sha256 under a recursive artifacts download tree."""
    candidates = list(artifacts_dir.rglob(f"{zip_name}.sha256"))
    if not candidates:
        raise SystemExit(
            f"missing {zip_name}.sha256 under {artifacts_dir} "
            f"(found {len(list(artifacts_dir.rglob('*')))} files)"
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda p: len(p.parts))
    return parse_hash_file(candidates[0])


def build_manifest(version: str, hash_x64: str) -> dict:
    tag = f"{TAG_PREFIX}{version}"
    base = f"https://github.com/canact/canact/releases/download/{tag}"
    return {
        "version": version,
        "description": (
            "Probe an LLM and return host policy: max tools, edit format, "
            "XML fallback, JSON repair"
        ),
        "homepage": "https://github.com/canact/canact",
        "license": "MIT OR Apache-2.0",
        "architecture": {
            "64bit": {
                "url": f"{base}/{X64_ZIP}",
                "hash": hash_x64.lower(),
            },
        },
        "bin": "canact.exe",
        "checkver": {
            "url": "https://api.github.com/repos/canact/canact/releases/latest",
            "jsonpath": "$.tag_name",
            "regex": r"v([\d.]+)",
        },
        "autoupdate": {
            "architecture": {
                "64bit": {
                    "url": (
                        "https://github.com/canact/canact/releases/download/"
                        "v$version/" + X64_ZIP
                    ),
                    "hash": {
                        "url": "$url.sha256",
                        "regex": r"^([a-fA-F0-9]{64})",
                    },
                },
            }
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version",
        required=True,
        help="Semver without tag prefix, e.g. 0.1.1",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory with cargo-dist artifacts (searched for *.zip.sha256)",
    )
    p.add_argument("--hash-x64", help="SHA256 of the x86_64 Windows zip")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write bucket/canact.json",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 only if existing --output matches generated content",
    )
    args = p.parse_args()

    version = args.version
    if version.startswith("canact-v"):
        version = version[len("canact-v") :]
    elif version.startswith("v"):
        version = version[1:]

    if args.artifacts_dir is not None:
        hash_x64 = find_hash(args.artifacts_dir, X64_ZIP)
    elif args.hash_x64 is not None:
        hash_x64 = args.hash_x64
    else:
        raise SystemExit("error: provide --artifacts-dir or --hash-x64")

    manifest = build_manifest(version, hash_x64)
    text = json.dumps(manifest, indent=4) + "\n"

    if args.check:
        if not args.output.is_file():
            print(f"missing {args.output}", file=sys.stderr)
            return 1
        existing = args.output.read_text(encoding="utf-8")
        if existing != text:
            print(f"{args.output} is stale for version {version}", file=sys.stderr)
            return 1
        print(f"ok: {args.output} matches {version}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output} for version {version}")
    print(f"  64bit  {hash_x64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
