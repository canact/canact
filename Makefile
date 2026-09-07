.PHONY: help check brand python-test scoop-manifest-test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

brand: ## Rasterize docs/brand/canact.svg into /tmp/canact-brand
	bash scripts/render-brand.sh

python-test: ## FOSSA filter and Scoop generator tests
	python3 scripts/test_fossa_filter.py
	python3 scripts/test_update_scoop_manifest.py

scoop-manifest-test: ## Scoop manifest generator tests
	python3 scripts/test_update_scoop_manifest.py

check: ## fmt, clippy, test, deny (same as CI lint+test)
	cargo fmt --check
	RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -- -D warnings
	RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets --features runtime -- -D warnings
	RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets --features cli -- -D warnings
	RUSTFLAGS="-D warnings" cargo test --locked
	RUSTFLAGS="-D warnings" cargo test --locked --features runtime
	RUSTFLAGS="-D warnings" cargo test --locked --features cli
	bash scripts/deny-check.sh
	python3 scripts/test_fossa_filter.py
	python3 scripts/test_update_scoop_manifest.py
