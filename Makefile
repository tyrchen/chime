DIST_DIR ?= $(CURDIR)/dist
APP_BUNDLE := $(DIST_DIR)/Chime.app
CARGO_TARGET_DIR := $(shell cargo metadata --format-version 1 --no-deps | jq -r .target_directory)

.PHONY: build test lint fmt eval swift-build macos-bundle install-macos clean

build:
	@cargo build --workspace --all-targets

test:
	@cargo test --workspace --all-features

lint:
	@cargo +nightly fmt --all -- --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt:
	@cargo +nightly fmt --all

eval:
	@cargo run --release -p chime-cli -- eval

swift-build:
	@cargo build --release -p chime-ffi
	@cd macos && swift build -c release -Xlinker -L$(CARGO_TARGET_DIR)/release

macos-bundle: swift-build
	@rm -rf "$(APP_BUNDLE)"
	@mkdir -p "$(APP_BUNDLE)/Contents/MacOS" "$(APP_BUNDLE)/Contents/Resources"
	@cp macos/.build/release/ChimeIME "$(APP_BUNDLE)/Contents/MacOS/Chime"
	@cp macos/Resources/Info.plist "$(APP_BUNDLE)/Contents/Info.plist"
	@codesign --force --deep --sign - "$(APP_BUNDLE)"
	@echo "Built $(APP_BUNDLE)"

install-macos: macos-bundle
	@mkdir -p "$(HOME)/Library/Input Methods"
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app"
	@cp -R "$(APP_BUNDLE)" "$(HOME)/Library/Input Methods/Chime.app"
	@echo "Installed to ~/Library/Input Methods/Chime.app"

clean:
	@cargo clean
	@rm -rf "$(DIST_DIR)" macos/.build
