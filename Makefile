.PHONY: help check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

check: ## fmt, clippy, test, deny (same as CI lint+test)
	cargo fmt --check
	RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -- -D warnings
	RUSTFLAGS="-D warnings" cargo test --locked
	bash scripts/deny-check.sh
