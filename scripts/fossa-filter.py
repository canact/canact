#!/usr/bin/env python3
"""Filter known false positives from FOSSA test JSON output.

Exit 0 if all issues are documented false positives.
Exit 1 if any genuine issues remain after filtering.
"""

import json
import sys

# Crates with "apache-2.0 WITH llvm-exception" that FOSSA cannot parse.
# The LLVM exception permits compilation without license propagation.
LLVM_EXCEPTION_CRATES = {
    "cc",
    "compiler_builtins",
    "wasi",
    "wasip2",
    "wit-bindgen-core",
    "wit-bindgen-rust",
    "wit-bindgen-rust-macro",
}

LLVM_EXCEPTION_LICENSES = {
    "apache-2.0 WITH llvm-exception",
    "Apache-2.0 WITH LLVM-exception",
}


def is_false_positive(pkg: str, license_id: str, issue_type: str = "") -> bool:
    """Return True if the (package, license, type) tuple is a documented FP."""
    del issue_type
    if pkg in LLVM_EXCEPTION_CRATES and license_id in LLVM_EXCEPTION_LICENSES:
        return True
    if pkg == "r-efi" and license_id in ("LGPL-2.1-or-later", "LGPL-2.1+"):
        return True
    if pkg == "ring" and "ssleay" in license_id.lower():
        return True
    if pkg == "aws-lc-sys" and license_id.startswith("GPL-"):
        return True
    if pkg == "security-framework" and license_id == "APSL-2.0":
        return True
    return False


def extract_package(issue: dict) -> str:
    """Extract the package name from a FOSSA issue.

    FOSSA uses 'revisionId' with format 'cargo+cc$1.0.106'.
    """
    rev = issue.get("revisionId", "")
    if rev:
        if "+" in rev:
            rev = rev.split("+", 1)[1]
        if "$" in rev:
            rev = rev.rsplit("$", 1)[0]
        return rev
    return issue.get("package", "") or issue.get("name", "") or ""


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: fossa-filter.py <fossa-results.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        data = json.load(f)

    if isinstance(data, list):
        issues = data
    else:
        issues = data.get("issues", data.get("issue", []))
    if not isinstance(issues, list):
        issues = []

    real_issues = []
    filtered_count = 0

    for issue in issues:
        pkg = extract_package(issue)
        lic = issue.get("license", "") or issue.get("licenseId", "") or ""
        itype = issue.get("type", "") or issue.get("issueType", "") or ""

        if is_false_positive(pkg, lic, itype):
            filtered_count += 1
            continue

        real_issues.append(issue)

    if real_issues:
        print(
            f"FAIL: {len(real_issues)} genuine issue(s) after filtering "
            f"{filtered_count} known false positives:"
        )
        for r in real_issues:
            pkg = extract_package(r)
            lic = r.get("license", "") or r.get("licenseId", "?")
            itype = r.get("type", "") or r.get("issueType", "?")
            print(f"  - {pkg}  {lic}  ({itype})")
        return 1

    print(f"OK: All {filtered_count} issue(s) are documented false positives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
