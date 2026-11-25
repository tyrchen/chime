DIST_DIR ?= $(CURDIR)/dist
APP_BUNDLE := $(DIST_DIR)/Chime.app
CORE_DICT := $(DIST_DIR)/core.imedict
NGRAMS_TSV := $(DIST_DIR)/ngrams.tsv
BASE_NEURAL_DIR := $(DIST_DIR)/base_neural
ICONSET_DIR := $(DIST_DIR)/AppIcon.iconset
MENU_ICON_PDF := $(DIST_DIR)/menu_icon.pdf
APP_ICON_ICNS := $(DIST_DIR)/AppIcon.icns
CARGO_TARGET_DIR := $(shell cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
JIEBA_DICT := $(shell ls -1 $(HOME)/.cargo/registry/src/*/jieba-rs-*/src/data/dict.txt 2>/dev/null | tail -n 1)
LSREGISTER := /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

.PHONY: build test lint fmt build-models build-core-dict build-icons eval swift-build macos-bundle install-macos clean

build:
	@cargo build --workspace --all-targets

test:
	@cargo test --workspace --all-features

lint:
	@cargo +nightly fmt --all -- --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt:
	@cargo +nightly fmt --all

build-models:
	@if [ ! -f "$(BASE_NEURAL_DIR)/model.safetensors" ] || [ ! -f "$(NGRAMS_TSV)" ]; then \
		uv run --python 3.12 --no-project --with torch --with transformers --with huggingface_hub --with safetensors \
			python -I tools/distill_qwen_base.py --out-neural "$(BASE_NEURAL_DIR)" --out-ngrams "$(NGRAMS_TSV)"; \
	fi

build-core-dict: build-models
	@mkdir -p "$(DIST_DIR)"
	@if [ -n "$(JIEBA_DICT)" ] && [ -f "$(JIEBA_DICT)" ]; then \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --jieba-dict "$(JIEBA_DICT)" --ngrams "$(NGRAMS_TSV)" --min-freq 20; \
	else \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --ngrams "$(NGRAMS_TSV)"; \
	fi

build-icons:
	@mkdir -p "$(DIST_DIR)"
	@swift macos/Scripts/generate_icons.swift "$(MENU_ICON_PDF)" "$(ICONSET_DIR)"
	@iconutil -c icns "$(ICONSET_DIR)" -o "$(APP_ICON_ICNS)"

eval: build-core-dict
	@cargo run --release -p chime-cli -- eval --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)"

swift-build:
	@cargo build --release -p chime-ffi
	@cd macos && swift build -c release -Xlinker -L$(CARGO_TARGET_DIR)/release -Xlinker -w

macos-bundle: build-core-dict build-icons swift-build
	@rm -rf "$(APP_BUNDLE)"
	@mkdir -p "$(APP_BUNDLE)/Contents/MacOS" "$(APP_BUNDLE)/Contents/Resources"
	@cp macos/.build/release/ChimeIME "$(APP_BUNDLE)/Contents/MacOS/Chime"
	@cp macos/Resources/Info.plist "$(APP_BUNDLE)/Contents/Info.plist"
	@cp -R macos/Resources/en.lproj "$(APP_BUNDLE)/Contents/Resources/en.lproj"
	@cp -R macos/Resources/zh-Hans.lproj "$(APP_BUNDLE)/Contents/Resources/zh-Hans.lproj"
	@cp -R macos/Resources/zh_CN.lproj "$(APP_BUNDLE)/Contents/Resources/zh_CN.lproj"
	@cp "$(MENU_ICON_PDF)" "$(APP_BUNDLE)/Contents/Resources/menu_icon.pdf"
	@cp "$(APP_ICON_ICNS)" "$(APP_BUNDLE)/Contents/Resources/AppIcon.icns"
	@cp "$(CORE_DICT)" "$(APP_BUNDLE)/Contents/Resources/core.imedict"
	@cp -R "$(BASE_NEURAL_DIR)" "$(APP_BUNDLE)/Contents/Resources/neural"
	@cp NOTICE "$(APP_BUNDLE)/Contents/Resources/NOTICE"
	@chmod -R u+rwX,go+rX "$(APP_BUNDLE)"
	@codesign --force --deep --sign - "$(APP_BUNDLE)"
	@echo "Built $(APP_BUNDLE) (core.imedict + Qwen2.5 MLX Metal neural base + localized icons)"

install-macos: macos-bundle
	@mkdir -p "$(HOME)/Library/Input Methods"
	@-pkill -x Chime 2>/dev/null || true
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app"
	@cp -R "$(APP_BUNDLE)" "$(HOME)/Library/Input Methods/Chime.app"
	@-"$(LSREGISTER)" -f -R -trusted "$(HOME)/Library/Input Methods/Chime.app"
	@swift macos/Scripts/register_ime.swift "$(HOME)/Library/Input Methods/Chime.app"
	@-killall TextInputMenuAgent TextInputSwitcher KeyboardSettings imklaunchagent 2>/dev/null || true
	@echo "Installed and registered Chime 拼音 at ~/Library/Input Methods/Chime.app"

clean:
	@cargo clean
	@rm -rf "$(DIST_DIR)" macos/.build
