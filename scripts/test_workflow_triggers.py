#!/usr/bin/env python3
"""Lock Recipe A and the cheap release-please split."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _on_block(text: str) -> str:
    start = text.index("\non:")
    rest = text[start + 1 :]
    end = rest.index("\njobs:")
    block = rest[:end]
    lines = []
    for line in block.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


class WorkflowTriggerTests(unittest.TestCase):
    def test_ci_has_no_push_compile(self) -> None:
        on_block = _on_block((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
        self.assertIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotIn("tags:", on_block)

    def test_release_is_tag_or_dispatch_only(self) -> None:
        on_block = _on_block((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
        self.assertNotIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("tags:", on_block)
        self.assertNotIn("branches:", on_block)

    def test_release_please_is_main_only(self) -> None:
        text = (WORKFLOWS / "release-please.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("push:", on_block)
        self.assertIn("branches: [main]", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")

    def test_auto_merge_skips_release_please_head(self) -> None:
        text = (WORKFLOWS / "auto-approve.yml").read_text(encoding="utf-8")
        self.assertIn("!startsWith(github.head_ref, 'release-please')", text)
        self.assertIn("autorelease: pending", text)

    def test_cheap_pr_status_checks_do_not_cancel(self) -> None:
        for name in ("pr-title.yml", "dco.yml"):
            text = (WORKFLOWS / name).read_text(encoding="utf-8")
            self.assertIn("cancel-in-progress: false", text, name)

    def test_rerun_cancelled_checks_is_check_run_only(self) -> None:
        text = (WORKFLOWS / "rerun-cancelled-pr-checks.yml").read_text(
            encoding="utf-8"
        )
        on_block = _on_block(text)
        self.assertIn("check_run:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")
        self.assertIn("cancel-in-progress: false", text)

    def test_publish_crates_is_tag_or_dispatch(self) -> None:
        text = (WORKFLOWS / "publish-crates.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("tags:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotIn("branches:", on_block)
        self.assertIn("id-token: write", text)
        self.assertIn("crates-io-auth-action", text)
        self.assertNotIn("CARGO_REGISTRY_TOKEN: ${{ secrets.", text)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")
        # Dispatch of an older tag must not run the wrapper from that
        # tree (v0.1.2 has no scripts/publish-crates.sh).
        self.assertIn("path: publisher", text)
        self.assertIn("path: crate", text)
        self.assertIn("ref: ${{ github.sha }}", text)
        self.assertIn("ref: ${{ inputs.tag || github.ref }}", text)
        self.assertIn("working-directory: crate", text)
        self.assertIn("bash ../publisher/scripts/publish-crates.sh", text)
        self.assertNotIn("run: bash scripts/publish-crates.sh", text)

    def test_apply_release_notes_is_dispatch_only(self) -> None:
        text = (WORKFLOWS / "apply-release-notes.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertNotIn("pull_request:", on_block)
        self.assertNotIn("push:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")

    def test_rust_release_type_is_minor_pre_major(self) -> None:
        cfg = (ROOT / "release-please-config.json").read_text(encoding="utf-8")
        self.assertIn('"release-type": "rust"', cfg)
        self.assertIn('"bump-minor-pre-major": true', cfg)
        self.assertIn('"include-component-in-tag": false', cfg)
        self.assertNotIn("bump-patch-for-minor-pre-major", cfg)
        dist = (ROOT / "dist-workspace.toml").read_text(encoding="utf-8")
        self.assertIn('pr-run-mode = "skip"', dist)

    def test_required_jobs_stay_named_on_release_please(self) -> None:
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        sec = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
        self.assertIn("name: Lint", ci)
        self.assertIn("name: Test", ci)
        self.assertIn("ubuntu-latest, macos-latest, windows-latest", ci)
        self.assertIn("name: CodeQL (${{ matrix.language }})", sec)
        # Job-level skip would drop the required check name.
        lint = ci[ci.index("name: Lint") : ci.index("name: Test")]
        test = ci[ci.index("name: Test") : ci.index("name: Fuzz smoke")]
        self.assertNotIn("startsWith(github.head_ref, 'release-please')", lint.split("steps:")[0])
        self.assertNotIn("startsWith(github.head_ref, 'release-please')", test.split("steps:")[0])
        codeql = sec[sec.index("name: CodeQL") :]
        self.assertNotIn(
            "startsWith(github.head_ref, 'release-please')",
            codeql.split("steps:")[0],
        )

    def test_release_please_uses_cargo_check_not_full_matrix(self) -> None:
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("cargo check --locked --all-targets", ci)
        self.assertGreaterEqual(ci.count("Release-please check"), 2)
        for cmd in (
            "cargo clippy --locked --all-targets",
            "cargo nextest run --locked",
            "cargo test --locked --doc",
            "cargo doc --locked --no-deps --all-features",
        ):
            idx = ci.index(cmd)
            window = ci[max(0, idx - 400) : idx]
            self.assertIn("!startsWith(github.head_ref, 'release-please')", window, cmd)

    def test_codeql_standin_on_release_please(self) -> None:
        sec = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
        self.assertIn("startsWith(github.head_ref, 'release-please')", sec)
        for pin in (
            "github/codeql-action/init@",
            "github/codeql-action/analyze@",
        ):
            idx = sec.index(pin)
            window = sec[max(0, idx - 400) : idx]
            self.assertIn("!startsWith(github.head_ref, 'release-please')", window, pin)

    def test_docs_is_cheap_pages_promote(self) -> None:
        text = (WORKFLOWS / "docs.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("pull_request:", on_block)
        self.assertIn("push:", on_block)
        self.assertNotIn("merge_group:", on_block)
        self.assertNotRegex(text, r"cargo (test|nextest|clippy|fuzz)")
        self.assertIn("mdbook build", text)
        self.assertIn("actions/deploy-pages@", text)

    def test_tag_update_does_not_republish(self) -> None:
        rel = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
        crates = (WORKFLOWS / "publish-crates.yml").read_text(encoding="utf-8")
        self.assertIn("github.event.created", rel)
        self.assertIn("github.event.created", crates)
        self.assertIn("needs.plan.result == 'success'", rel)

    def test_lint_runs_makefile_python_tests(self) -> None:
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        lint = ci[ci.index("name: Lint") : ci.index("name: Test")]
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        start = makefile.index("python-test:")
        end = makefile.index("\nscoop-manifest-test:", start)
        scripts = [
            line.strip()
            for line in makefile[start:end].splitlines()
            if line.strip().startswith("python3 scripts/test_")
        ]
        self.assertIn("python3 scripts/test_sign_git_tag.py", scripts)
        self.assertIn("python3 scripts/test_publish_crates.py", scripts)
        check = makefile[makefile.index("check:") :]
        for script in scripts:
            self.assertIn(script, lint, script)
            self.assertIn(f"\t{script}", check, script)

    def test_sign_tags_is_gpg_not_cosign(self) -> None:
        text = (WORKFLOWS / "sign-tags.yml").read_text(encoding="utf-8")
        on_block = _on_block(text)
        self.assertIn("workflow_dispatch:", on_block)
        self.assertIn("release:", on_block)
        self.assertNotIn("cosign-installer", text)
        self.assertIn("ghaction-import-gpg@", text)
        self.assertIn("scripts/sign-git-tag.sh", text)

    def test_fossa_push_and_pr_share_cargo_path_filter(self) -> None:
        on_block = _on_block((WORKFLOWS / "fossa.yml").read_text(encoding="utf-8"))
        self.assertIn("push:", on_block)
        self.assertIn("pull_request:", on_block)
        self.assertIn("workflow_dispatch:", on_block)
        push = on_block[on_block.index("push:") : on_block.index("pull_request:")]
        pr = on_block[on_block.index("pull_request:") : on_block.index("workflow_dispatch:")]
        for needle in (
            "Cargo.*",
            "src/**",
            "scripts/fossa-filter.py",
            "scripts/test_fossa_filter.py",
            ".github/workflows/fossa.yml",
        ):
            self.assertIn(needle, push, needle)
            self.assertIn(needle, pr, needle)

    def test_scheduled_jobs_report_failures(self) -> None:
        cases = (
            (
                "security.yml",
                "CodeQL red",
                "JOB_RESULTS: codeql=${{ needs.codeql.result }}",
            ),
            (
                "scorecard.yml",
                "Scorecard red",
                "JOB_RESULTS: scorecard=${{ needs.scorecard.result }}",
            ),
            (
                "link-check.yml",
                "Link check red",
                "JOB_RESULTS: check=${{ needs.check.result }}",
            ),
        )
        reporter_if = (
            "if: always() && (github.event_name == 'schedule' "
            "|| github.event_name == 'workflow_dispatch')"
        )
        for name, prefix, job_results in cases:
            text = (WORKFLOWS / name).read_text(encoding="utf-8")
            reporter = text[text.index("report-failure") :]
            self.assertIn("scripts/report-scheduled-failure.py", reporter, name)
            self.assertIn(prefix, reporter, name)
            self.assertIn(job_results, reporter, name)
            self.assertIn("issues: write", reporter, name)
            self.assertIn("vars.NIGHTLY_FAILURE_ASSIGNEE || 'SebTardif'", reporter, name)
            self.assertIn(reporter_if, reporter, name)
            self.assertIn("--label nightly-failure", reporter, name)
            self.assertIn("--label ready", reporter, name)
            self.assertNotIn("pull_request", reporter.split("steps:")[0], name)
        security = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
        security_reporter = security[security.index("report-failure") :]
        self.assertNotIn("dependency-review", security_reporter)
        stale = (WORKFLOWS / "stale.yml").read_text(encoding="utf-8")
        exempt = stale[stale.index("exempt-issue-labels") :]
        self.assertIn("nightly-failure", exempt)

    def test_release_sbom_and_nonfatal_provenance(self) -> None:
        rel = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
        sbom = rel[rel.index("\n  sbom:") : rel.index("\n  provenance:")]
        self.assertIn("cargo install cargo-cyclonedx --version 0.5.9 --locked", sbom)
        self.assertIn("canact-sbom.cdx.json", sbom)
        self.assertIn("publisher/scripts/stage-cyclonedx-sbom.sh", sbom)
        self.assertIn("path: publisher", sbom)
        self.assertIn("path: source", sbom)
        self.assertIn("ref: ${{ needs.plan.outputs.tag }}", sbom)
        self.assertIn('toolchain: "1.95"', sbom)
        self.assertNotIn("continue-on-error", sbom)
        provenance = rel[rel.index("\n  provenance:") : rel.index("\n  publish-homebrew-formula:")]
        self.assertGreaterEqual(provenance.count("continue-on-error: true"), 3)
        attest = provenance[provenance.index("Attest build provenance") : provenance.index("Install Cosign")]
        self.assertIn("continue-on-error: true", attest)
        sign = provenance[provenance.index("Sign and upload") :]
        self.assertIn("continue-on-error: true", sign)
        announce = rel[rel.index("\n  announce:") :]
        self.assertIn("- sbom", announce)
        self.assertIn("needs.provenance.result == 'failure'", announce)
        self.assertIn(
            "needs.sbom.result == 'skipped' || needs.sbom.result == 'success'",
            announce,
        )
        self.assertNotIn("needs.sbom.result == 'failure'", announce)
        sign_release = (WORKFLOWS / "sign-release.yml").read_text(encoding="utf-8")
        self.assertNotIn("continue-on-error", sign_release)
        self.assertIn("Attest build provenance", sign_release)
        self.assertIn("scripts/attach-release-signatures.sh", sign_release)


if __name__ == "__main__":
    unittest.main()
