DIST_DIR ?= $(CURDIR)/dist
APP_BUNDLE := $(DIST_DIR)/Chime.app
CORE_DICT := $(DIST_DIR)/core.imedict
CARGO_TARGET_DIR := $(shell cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
JIEBA_DICT := $(shell ls -1 $(HOME)/.cargo/registry/src/*/jieba-rs-*/src/data/dict.txt 2>/dev/null | tail -n 1)

.PHONY: build test lint fmt build-core-dict eval swift-build macos-bundle install-macos clean

build:
	@cargo build --workspace --all-targets

test:
	@cargo test --workspace --all-features

lint:
	@cargo +nightly fmt --all -- --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt:
	@cargo +nightly fmt --all

build-core-dict:
	@mkdir -p "$(DIST_DIR)"
	@if [ -n "$(JIEBA_DICT)" ] && [ -f "$(JIEBA_DICT)" ]; then \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --jieba-dict "$(JIEBA_DICT)" --min-freq 20; \
	else \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)"; \
	fi

eval: build-core-dict
	@cargo run --release -p chime-cli -- eval --dict "$(CORE_DICT)"

swift-build:
	@cargo build --release -p chime-ffi
	@cd macos && swift build -c release -Xlinker -L$(CARGO_TARGET_DIR)/release

macos-bundle: build-core-dict swift-build
	@rm -rf "$(APP_BUNDLE)"
	@mkdir -p "$(APP_BUNDLE)/Contents/MacOS" "$(APP_BUNDLE)/Contents/Resources"
	@cp macos/.build/release/ChimeIME "$(APP_BUNDLE)/Contents/MacOS/Chime"
	@cp macos/Resources/Info.plist "$(APP_BUNDLE)/Contents/Info.plist"
	@cp "$(CORE_DICT)" "$(APP_BUNDLE)/Contents/Resources/core.imedict"
	@cp NOTICE "$(APP_BUNDLE)/Contents/Resources/NOTICE"
	@codesign --force --deep --sign - "$(APP_BUNDLE)"
	@echo "Built $(APP_BUNDLE) with zero-copy mmap core.imedict"

install-macos: macos-bundle
	@mkdir -p "$(HOME)/Library/Input Methods"
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app"
	@cp -R "$(APP_BUNDLE)" "$(HOME)/Library/Input Methods/Chime.app"
	@echo "Installed to ~/Library/Input Methods/Chime.app"

clean:
	@cargo clean
	@rm -rf "$(DIST_DIR)" macos/.build
