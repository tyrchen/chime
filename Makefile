DIST_DIR ?= $(CURDIR)/dist
RELEASE_DIR := $(DIST_DIR)/releases
PACKS_BUILD_DIR := $(DIST_DIR)/packs
APP_BUNDLE := $(DIST_DIR)/Chime.app
CORE_DICT := $(DIST_DIR)/core.imedict
NGRAMS_TSV := $(DIST_DIR)/ngrams.tsv
MULTI_LEX_TSV := $(DIST_DIR)/multi_lexicon.tsv
CLASSICS_CORE_LEX := $(DIST_DIR)/classics_core_lexicon.tsv
CLASSICS_CORE_CONT := $(DIST_DIR)/classics_core_continuations.tsv
MERGED_CORE_LEX := $(DIST_DIR)/merged_core_lexicon.tsv
BASE_NEURAL_DIR := $(DIST_DIR)/base_neural
ICONSET_DIR := $(DIST_DIR)/AppIcon.iconset
MENU_ICON_PDF := $(DIST_DIR)/menu_icon.pdf
APP_ICON_ICNS := $(DIST_DIR)/AppIcon.icns
VERSION := $(shell cargo metadata --format-version 1 --no-deps | jq -r '.packages[0].version')
CARGO_TARGET_DIR := $(shell cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
JIEBA_DICT := $(shell ls -1 $(HOME)/.cargo/registry/src/*/jieba-rs-*/src/data/dict.txt 2>/dev/null | tail -n 1)
LSREGISTER := /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
APP_SIGN_IDENTITY ?= $(shell security find-identity -v -p codesigning 2>/dev/null | sed -n 's/.*"\(Developer ID Application: .*\)"/\1/p' | head -1)
INSTALLER_SIGN_IDENTITY ?= $(shell security find-identity -v -p basic 2>/dev/null | sed -n 's/.*"\(Developer ID Installer: .*\)"/\1/p' | head -1)
ASC_KEYCHAIN_SERVICE ?= chime-asc-key
NOTARY_PROFILE ?=

define run-notarytool
@set -eu; \
if [ -n "$(NOTARY_PROFILE)" ]; then \
	xcrun notarytool submit "$(1)" --keychain-profile "$(NOTARY_PROFILE)" --wait; \
else \
	item=$$(security find-generic-password -s "$(ASC_KEYCHAIN_SERVICE)" 2>/dev/null) || { echo "Missing notary credentials: set NOTARY_PROFILE or ASC_KEYCHAIN_SERVICE"; exit 1; }; \
	key_id=$$(echo "$$item" | sed -n 's/.*"acct"<blob>="\(.*\)"/\1/p'); \
	issuer=$$(echo "$$item" | sed -n 's/.*"icmt"<blob>="\(.*\)"/\1/p'); \
	dir=$$(mktemp -d); trap 'rm -rf "$$dir"' EXIT; umask 077; \
	security find-generic-password -s "$(ASC_KEYCHAIN_SERVICE)" -w | base64 -D > "$$dir/AuthKey_$$key_id.p8"; \
	xcrun notarytool submit "$(1)" --key "$$dir/AuthKey_$$key_id.p8" --key-id "$$key_id" --issuer "$$issuer" --wait; \
fi
endef

.PHONY: build test lint fmt build-models build-classics-corpus build-core-dict build-icons eval swift-build macos-bundle build-public-packs release-artifacts release-notarize install-macos paper paper-arxiv clean

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
	@if [ ! -f "$(BASE_NEURAL_DIR)/model.safetensors" ] || [ ! -f "$(NGRAMS_TSV)" ] || [ ! -f "$(MULTI_LEX_TSV)" ]; then \
		uv run --python 3.12 --no-project --with torch --with transformers --with huggingface_hub --with safetensors --with wordfreq --with pypinyin --with mlx \
			python -I tools/distill_qwen_base.py --out-neural "$(BASE_NEURAL_DIR)" --out-ngrams "$(NGRAMS_TSV)" --out-lexicon "$(MULTI_LEX_TSV)"; \
	fi

build-classics-corpus:
	@if [ ! -f "$(CLASSICS_CORE_LEX)" ] || [ ! -f "$(CLASSICS_CORE_CONT)" ]; then \
		uv run --python 3.12 --no-project --with opencc-python-reimplemented --with pypinyin \
			python -I tools/build_classics_corpus.py --dist-dir "$(DIST_DIR)"; \
	fi

build-core-dict: build-models build-classics-corpus
	@mkdir -p "$(DIST_DIR)"
	@cat "$(MULTI_LEX_TSV)" "$(CLASSICS_CORE_LEX)" > "$(MERGED_CORE_LEX)"
	@if [ -n "$(JIEBA_DICT)" ] && [ -f "$(JIEBA_DICT)" ]; then \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --jieba-dict "$(JIEBA_DICT)" --extra-tsv "$(MERGED_CORE_LEX)" --continuations "$(CLASSICS_CORE_CONT)" --ngrams "$(NGRAMS_TSV)" --min-freq 2; \
	else \
		cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --extra-tsv "$(MERGED_CORE_LEX)" --continuations "$(CLASSICS_CORE_CONT)" --ngrams "$(NGRAMS_TSV)"; \
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
	@xattr -cr "$(APP_BUNDLE)"
	@if [ -n "$(APP_SIGN_IDENTITY)" ]; then \
		echo "Signing $(APP_BUNDLE) with $(APP_SIGN_IDENTITY) (Hardened Runtime + Timestamp)..."; \
		codesign --force --options runtime --timestamp --sign "$(APP_SIGN_IDENTITY)" "$(APP_BUNDLE)/Contents/MacOS/Chime"; \
		codesign --force --options runtime --timestamp --sign "$(APP_SIGN_IDENTITY)" "$(APP_BUNDLE)"; \
		codesign --verify --deep --strict --verbose=2 "$(APP_BUNDLE)"; \
		! codesign -d --entitlements - "$(APP_BUNDLE)" 2>/dev/null | grep -q get-task-allow || { echo "Refusing get-task-allow entitlement in Developer ID bundle"; exit 1; }; \
	else \
		codesign --force --deep --sign - "$(APP_BUNDLE)"; \
	fi
	@echo "Built $(APP_BUNDLE) v$(VERSION) (core.imedict + Qwen3 6L 4-bit MLX Metal neural base)"

build-public-packs: build-classics-corpus
	@mkdir -p "$(PACKS_BUILD_DIR)" "$(RELEASE_DIR)"
	@if [ ! -f "$(DIST_DIR)/tech_pack_lexicon.tsv" ]; then \
		uv run --python 3.12 --no-project --with jieba --with pypinyin \
			python -I tools/prepare_pack_corpora.py --dist-dir "$(DIST_DIR)"; \
	fi
	@rm -rf "$(PACKS_BUILD_DIR)/classics-poetry.imepack" "$(PACKS_BUILD_DIR)/tech-developer.imepack"
	@cargo run --release -p chime-cli -- pack build "$(PACKS_BUILD_DIR)/classics-poetry.imepack" \
		--id classics-poetry \
		--name "古典诗词曲赋与成语全库" \
		--description "收录唐诗、宋词、元曲、诗经、楚辞、四书、古文名篇及新华成语歇后语共 26.6 万词条与 23.1 万条全篇 Tab 续写链，含 Rank-8 古汉语 LoRA" \
		--routing classical \
		--lora-scale 1.0 \
		--tsv "$(DIST_DIR)/classics_pack_lexicon.tsv" \
		--continuations "$(DIST_DIR)/classics_pack_continuations.tsv" \
		$(if $(wildcard $(DIST_DIR)/classics_lora.safetensors),--lora "$(DIST_DIR)/classics_lora.safetensors",)
	@cargo run --release -p chime-cli -- pack build "$(PACKS_BUILD_DIR)/tech-developer.imepack" \
		--id tech-developer \
		--name "开发者中英技术双语包" \
		--description "面向系统编程、Rust/Swift、端侧 AI 与云原生的中英双语术语库及 Rank-8 技术语境 LoRA，在终端与 IDE 中自动满权激活" \
		--routing developer \
		--lora-scale 1.0 \
		--tsv "$(DIST_DIR)/tech_pack_lexicon.tsv" \
		$(if $(wildcard $(DIST_DIR)/tech_lora.safetensors),--lora "$(DIST_DIR)/tech_lora.safetensors",)
	@/usr/bin/ditto -c -k --keepParent "$(PACKS_BUILD_DIR)/classics-poetry.imepack" "$(RELEASE_DIR)/classics-poetry.imepack.zip"
	@/usr/bin/ditto -c -k --keepParent "$(PACKS_BUILD_DIR)/tech-developer.imepack" "$(RELEASE_DIR)/tech-developer.imepack.zip"

release-artifacts: macos-bundle build-public-packs
	@mkdir -p "$(RELEASE_DIR)"
	@chmod +x macos/Scripts/pkg_scripts/postinstall
	@/usr/bin/pkgbuild \
		--component "$(APP_BUNDLE)" \
		--install-location "/Library/Input Methods" \
		--scripts macos/Scripts/pkg_scripts \
		--identifier "app.tyrchen.inputmethod.Chime.pkg" \
		--version "$(VERSION)" \
		$(if $(INSTALLER_SIGN_IDENTITY),--sign "$(INSTALLER_SIGN_IDENTITY)",) \
		"$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@if [ -n "$(INSTALLER_SIGN_IDENTITY)" ]; then \
		pkgutil --check-signature "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"; \
	fi
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --keepParent "$(APP_BUNDLE)" "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@cd "$(RELEASE_DIR)" && shasum -a 256 \
		"Chime-v$(VERSION)-macos-arm64.pkg" \
		"Chime-v$(VERSION)-macos-arm64.zip" \
		"classics-poetry.imepack.zip" \
		"tech-developer.imepack.zip" > SHA256SUMS.txt
	@echo "Packaged release artifacts in $(RELEASE_DIR):"
	@ls -lh "$(RELEASE_DIR)"

release-notarize: macos-bundle build-public-packs
	@test -n "$(APP_SIGN_IDENTITY)" || { echo "Missing Developer ID Application identity in keychain"; exit 1; }
	@test -n "$(INSTALLER_SIGN_IDENTITY)" || { echo "Missing Developer ID Installer identity in keychain"; exit 1; }
	@mkdir -p "$(RELEASE_DIR)"
	@echo "1/4 Submitting Chime.app to Apple Notary Service..."
	@rm -f "$(DIST_DIR)/Chime-notarize-app.zip"
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --keepParent "$(APP_BUNDLE)" "$(DIST_DIR)/Chime-notarize-app.zip"
	$(call run-notarytool,$(DIST_DIR)/Chime-notarize-app.zip)
	@rm -f "$(DIST_DIR)/Chime-notarize-app.zip"
	@echo "2/4 Stapling notarization ticket to Chime.app and packaging .zip..."
	@xcrun stapler staple "$(APP_BUNDLE)"
	@xcrun stapler validate "$(APP_BUNDLE)"
	@spctl --assess --type exec --verbose=4 "$(APP_BUNDLE)"
	@rm -f "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --keepParent "$(APP_BUNDLE)" "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@echo "3/4 Building Developer ID signed .pkg installer and submitting to Apple Notary Service..."
	@chmod +x macos/Scripts/pkg_scripts/postinstall
	@rm -f "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@/usr/bin/pkgbuild \
		--component "$(APP_BUNDLE)" \
		--install-location "/Library/Input Methods" \
		--scripts macos/Scripts/pkg_scripts \
		--identifier "app.tyrchen.inputmethod.Chime.pkg" \
		--version "$(VERSION)" \
		--sign "$(INSTALLER_SIGN_IDENTITY)" \
		"$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@pkgutil --check-signature "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	$(call run-notarytool,$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg)
	@echo "4/4 Stapling notarization ticket to .pkg and running Gatekeeper assessment..."
	@xcrun stapler staple "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@xcrun stapler validate "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@spctl --assess --type install --verbose=4 "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@cd "$(RELEASE_DIR)" && shasum -a 256 \
		"Chime-v$(VERSION)-macos-arm64.pkg" \
		"Chime-v$(VERSION)-macos-arm64.zip" \
		"classics-poetry.imepack.zip" \
		"tech-developer.imepack.zip" > SHA256SUMS.txt
	@echo "Successfully signed, notarized, and stapled all Chime v$(VERSION) release artifacts!"
	@cat "$(RELEASE_DIR)/SHA256SUMS.txt"

install-macos: macos-bundle
	@mkdir -p "$(HOME)/Library/Input Methods"
	@-pkill -x Chime 2>/dev/null || true
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app"
	@cp -R "$(APP_BUNDLE)" "$(HOME)/Library/Input Methods/Chime.app"
	@-"$(LSREGISTER)" -f -R -trusted "$(HOME)/Library/Input Methods/Chime.app"
	@-killall TextInputMenuAgent imklaunchagent 2>/dev/null || true
	@-killall -9 TextInputSwitcher CursorUIViewService 2>/dev/null || true
	@sleep 1
	@swift macos/Scripts/register_ime.swift "$(HOME)/Library/Input Methods/Chime.app"
	@echo "Installed and registered Chime 拼音 v$(VERSION) at ~/Library/Input Methods/Chime.app"

paper:
	@typst compile paper/chime-paper.typ paper/chime-paper.pdf
	@cp paper/chime-paper.pdf site/chime-paper.pdf
	@echo "Compiled paper/chime-paper.pdf and copied to site/chime-paper.pdf"

paper-arxiv: paper
	@printf '#import "figures.typ": fig-architecture\n#set page(width: 7.1in, height: auto, margin: 2pt)\n#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Songti SC"), size: 9.6pt)\n#fig-architecture()\n' > paper/_export_fig1.typ
	@printf '#import "figures.typ": fig-scaling-and-transfer\n#set page(width: 7.1in, height: auto, margin: 2pt)\n#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Songti SC"), size: 9.6pt)\n#fig-scaling-and-transfer()\n' > paper/_export_fig2.typ
	@typst compile paper/_export_fig1.typ paper/arxiv/fig1_architecture.pdf
	@typst compile paper/_export_fig2.typ paper/arxiv/fig2_scaling.pdf
	@rm -f paper/_export_fig1.typ paper/_export_fig2.typ
	@cd paper/arxiv && COPYFILE_DISABLE=1 tar -czf chime-arxiv-source.tar.gz 00README.json main.tex main.bbl fig1_architecture.pdf fig2_scaling.pdf
	@echo "Packaged paper/arxiv/chime-arxiv-source.tar.gz"

clean:
	@cargo clean
	@rm -rf "$(DIST_DIR)" macos/.build
