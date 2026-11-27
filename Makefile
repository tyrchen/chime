DIST_DIR ?= $(CURDIR)/dist
RELEASE_DIR := $(DIST_DIR)/releases
PACKS_BUILD_DIR := $(DIST_DIR)/packs
APP_BUNDLE := $(DIST_DIR)/Chime.app
VOICE_CACHE ?= $(DIST_DIR)/voice-cache
CORE_DICT := $(DIST_DIR)/core.imedict
NGRAMS_TSV := $(DIST_DIR)/ngrams.tsv
MULTI_LEX_TSV := $(DIST_DIR)/multi_lexicon.tsv
CLASSICS_CORE_LEX := $(DIST_DIR)/classics_core_lexicon.tsv
CLASSICS_CORE_CONT := $(DIST_DIR)/classics_core_continuations.tsv
MERGED_CORE_LEX := $(DIST_DIR)/merged_core_lexicon.tsv
BASE_NEURAL_DIR := $(DIST_DIR)/base_neural
MLX_METAL_OUT_DIR := /tmp/chime-mlx-metallib
ICONSET_DIR := $(DIST_DIR)/AppIcon.iconset
MENU_ICON_PDF := $(DIST_DIR)/menu_icon.pdf
APP_ICON_ICNS := $(DIST_DIR)/AppIcon.icns
MACOSX_DEPLOYMENT_TARGET := 14.0
export MACOSX_DEPLOYMENT_TARGET
CMAKE_OSX_DEPLOYMENT_TARGET := 14.0
export CMAKE_OSX_DEPLOYMENT_TARGET
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
	key_id=$$(printf '%s\n' "$$item" | sed -n 's/.*"acct"<blob>="\(.*\)"/\1/p'); \
	issuer=$$(printf '%s\n' "$$item" | sed -n 's/.*"icmt"<blob>="\(.*\)"/\1/p'); \
	dir=$$(mktemp -d); trap 'rm -rf "$$dir"' EXIT; umask 077; \
	security find-generic-password -s "$(ASC_KEYCHAIN_SERVICE)" -w | base64 -D > "$$dir/AuthKey_$$key_id.p8"; \
	xcrun notarytool submit "$(1)" --key "$$dir/AuthKey_$$key_id.p8" --key-id "$$key_id" --issuer "$$issuer" --wait; \
fi
endef

.PHONY: build test lint fmt build-models build-classics-corpus build-core-dict build-icons eval bench bench-dev bench-test bench-dev-10k bench-test-10k swift-build macos-bundle build-public-packs release-artifacts release-notarize install-macos paper paper-arxiv clean

build:
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo build --workspace --all-targets

test:
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo test --workspace --all-features

lint:
	@cargo +nightly fmt --all -- --check
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo clippy --workspace --all-targets --all-features -- -D warnings

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
	@if [ ! -f "$(MULTI_LEX_TSV)" ] || [ tools/calibrate_modern_lexicon.py -nt "$(MULTI_LEX_TSV)" ] || [ crates/lexicon/data/polyphones.yaml -nt "$(MULTI_LEX_TSV)" ]; then \
		uv run --python 3.12 --no-project --with torch --with transformers --with safetensors --with wordfreq --with pypinyin \
			python -I tools/calibrate_modern_lexicon.py --out "$(MULTI_LEX_TSV)" --polyphones-yaml crates/lexicon/data/polyphones.yaml; \
	fi
	@if [ ! -f "$(CORE_DICT)" ] || [ "$(MULTI_LEX_TSV)" -nt "$(CORE_DICT)" ] || [ crates/lexicon/data/closed_grammar.yaml -nt "$(CORE_DICT)" ] || [ crates/lexicon/data/english_casing.yaml -nt "$(CORE_DICT)" ] || [ crates/lexicon/data/polyphones.yaml -nt "$(CORE_DICT)" ] || [ crates/lexicon/src/base_builder.rs -nt "$(CORE_DICT)" ]; then \
		python3 -c 'import pathlib; d = pathlib.Path("$(DIST_DIR)"); m = (d/"multi_lexicon.tsv").read_text("utf-8"); c_lines = []; \
		[c_lines.append(f"{p[0]}\t{p[1]}\t{(float(p[2]) - 1.25 if len(p[0]) <= 4 else float(p[2])):.4f}") for ln in (d/"classics_core_lexicon.tsv").read_text("utf-8").splitlines() if "\t" in ln for p in [ln.split("\t")] if len(p) >= 3 and len(p[0]) >= 2]; \
		(d/"merged_core_lexicon.tsv").write_text(m + ("\n" if not m.endswith("\n") else "") + "\n".join(c_lines) + "\n", "utf-8")'; \
		if [ -n "$(JIEBA_DICT)" ] && [ -f "$(JIEBA_DICT)" ]; then \
			MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --jieba-dict "$(JIEBA_DICT)" --extra-tsv "$(MERGED_CORE_LEX)" --continuations "$(CLASSICS_CORE_CONT)" --ngrams "$(NGRAMS_TSV)" --min-freq 2; \
		else \
			MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- dict-build-base -o "$(CORE_DICT)" --extra-tsv "$(MERGED_CORE_LEX)" --continuations "$(CLASSICS_CORE_CONT)" --ngrams "$(NGRAMS_TSV)"; \
		fi; \
	fi

build-icons:
	@mkdir -p "$(DIST_DIR)"
	@swift macos/Scripts/generate_icons.swift "$(MENU_ICON_PDF)" "$(ICONSET_DIR)"
	@iconutil -c icns "$(ICONSET_DIR)" -o "$(APP_ICON_ICNS)"

eval: build-core-dict
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- eval --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)"

bench-dev: build-core-dict
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- bench --route-a --beam-width 8 benchmarks/dev_suite.tsv --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)" --record benchmarks/history.jsonl --tag "v$(VERSION)-route-a-dev"

bench-test: build-core-dict
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- bench --route-a --beam-width 8 benchmarks/test_holdout.tsv --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)" --record benchmarks/history.jsonl --tag "v$(VERSION)-route-a-holdout"

bench-dev-10k: build-core-dict
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- bench --route-a --beam-width 8 benchmarks/dev_suite_10k.tsv --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)" --record benchmarks/history.jsonl --tag "v$(VERSION)-route-a-dev10k"

bench-test-10k: build-core-dict
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- bench --route-a --beam-width 8 benchmarks/test_holdout_10k.tsv --dict "$(CORE_DICT)" --neural "$(BASE_NEURAL_DIR)" --record benchmarks/history.jsonl --tag "v$(VERSION)-route-a-holdout10k"

bench: bench-dev bench-test

swift-build:
	@mkdir -p "$(MLX_METAL_OUT_DIR)"
	@MACOSX_DEPLOYMENT_TARGET=14.0 CMAKE_OSX_DEPLOYMENT_TARGET=14.0 MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" RUSTFLAGS="--remap-path-prefix=$(HOME)=~" cargo build --release -p chime-ffi
	@test -f "$(MLX_METAL_OUT_DIR)/mlx.metallib" || { echo "Missing $(MLX_METAL_OUT_DIR)/mlx.metallib after chime-ffi build"; exit 1; }
	@cd macos && MACOSX_DEPLOYMENT_TARGET=14.0 swift build -c release -Xlinker -L$(CARGO_TARGET_DIR)/release -Xlinker -w

macos-bundle: build-core-dict build-icons swift-build
	@rm -rf "$(APP_BUNDLE)"
	@mkdir -p "$(APP_BUNDLE)/Contents/MacOS" "$(APP_BUNDLE)/Contents/Resources"
	@cp macos/.build/release/ChimeIME "$(APP_BUNDLE)/Contents/MacOS/Chime"
	@cp "$(MLX_METAL_OUT_DIR)/mlx.metallib" "$(APP_BUNDLE)/Contents/Resources/mlx.metallib"
	@cp macos/Resources/Info.plist "$(APP_BUNDLE)/Contents/Info.plist"
	@cp -R macos/Resources/en.lproj "$(APP_BUNDLE)/Contents/Resources/en.lproj"
	@cp -R macos/Resources/zh-Hans.lproj "$(APP_BUNDLE)/Contents/Resources/zh-Hans.lproj"
	@cp -R macos/Resources/zh_CN.lproj "$(APP_BUNDLE)/Contents/Resources/zh_CN.lproj"
	@cp "$(MENU_ICON_PDF)" "$(APP_BUNDLE)/Contents/Resources/menu_icon.pdf"
	@cp "$(APP_ICON_ICNS)" "$(APP_BUNDLE)/Contents/Resources/AppIcon.icns"
	@cp "$(CORE_DICT)" "$(APP_BUNDLE)/Contents/Resources/core.imedict"
	@cp -R "$(BASE_NEURAL_DIR)" "$(APP_BUNDLE)/Contents/Resources/neural"
	@python3 macos/Scripts/bundle_voice.py "$(APP_BUNDLE)/Contents" "$(VOICE_CACHE)"
	@cp NOTICE "$(APP_BUNDLE)/Contents/Resources/NOTICE"
	@chmod -R u+rwX,go+rX "$(APP_BUNDLE)"
	@xattr -cr "$(APP_BUNDLE)"
	@! otool -L "$(APP_BUNDLE)/Contents/MacOS/Chime" | grep -q "libswift_stdio" || { echo "FATAL: Chime still links macOS 26+/15.4+ only libswift_stdio.dylib"; exit 1; }
	@if [ -n "$(APP_SIGN_IDENTITY)" ]; then \
		echo "Signing $(APP_BUNDLE) with $(APP_SIGN_IDENTITY) (Hardened Runtime + Timestamp)..."; \
		set -e; for binary in "$(APP_BUNDLE)"/Contents/Frameworks/*.dylib "$(APP_BUNDLE)/Contents/Helpers/ChimeVoice"; do \
			codesign --force --options runtime --timestamp --sign "$(APP_SIGN_IDENTITY)" "$$binary"; \
		done; \
		codesign --force --options runtime --timestamp --entitlements macos/Resources/Chime.entitlements --sign "$(APP_SIGN_IDENTITY)" "$(APP_BUNDLE)/Contents/MacOS/Chime"; \
		codesign --force --options runtime --timestamp --entitlements macos/Resources/Chime.entitlements --sign "$(APP_SIGN_IDENTITY)" "$(APP_BUNDLE)"; \
		codesign --verify --deep --strict --verbose=2 "$(APP_BUNDLE)"; \
		! codesign -d --entitlements - "$(APP_BUNDLE)" 2>/dev/null | grep -q get-task-allow || { echo "Refusing get-task-allow entitlement in Developer ID bundle"; exit 1; }; \
	else \
		codesign --force --deep --entitlements macos/Resources/Chime.entitlements --sign - "$(APP_BUNDLE)"; \
	fi
	@echo "Built $(APP_BUNDLE) v$(VERSION) (macOS 14.0+ compatible, bundled mlx.metallib + Qwen3 28L 4-bit MLX Metal)"

build-public-packs: build-classics-corpus
	@mkdir -p "$(PACKS_BUILD_DIR)" "$(RELEASE_DIR)"
	@if [ ! -f "$(DIST_DIR)/tech_pack_lexicon.tsv" ]; then \
		uv run --python 3.12 --no-project --with jieba --with pypinyin \
			python -I tools/prepare_pack_corpora.py --dist-dir "$(DIST_DIR)"; \
	fi
	@rm -rf "$(PACKS_BUILD_DIR)/classics-poetry.imepack" "$(PACKS_BUILD_DIR)/tech-developer.imepack"
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- pack build "$(PACKS_BUILD_DIR)/classics-poetry.imepack" \
		--id classics-poetry \
		--name "古典诗词曲赋与成语全库" \
		--description "收录唐诗、宋词、元曲、诗经、楚辞、四书、古文名篇及新华成语歇后语共 26.6 万词条与 23.1 万条全篇 Tab 续写链，含 28L Rank-8 古汉语 LoRA" \
		--routing classical \
		--lora-scale 1.0 \
		--tsv "$(DIST_DIR)/classics_pack_lexicon.tsv" \
		--continuations "$(DIST_DIR)/classics_pack_continuations.tsv" \
		$(if $(wildcard $(DIST_DIR)/classics_lora.safetensors),--lora "$(DIST_DIR)/classics_lora.safetensors",)
	@MLX_RS_METAL_PATH="$(MLX_METAL_OUT_DIR)" cargo run --release -p chime-cli -- pack build "$(PACKS_BUILD_DIR)/tech-developer.imepack" \
		--id tech-developer \
		--name "开发者中英技术双语包" \
		--description "面向系统编程、Rust/Swift、端侧 AI 与云原生的中英双语术语库及 28L Rank-8 技术语境 LoRA，在终端与 IDE 中自动满权激活" \
		--routing developer \
		--lora-scale 1.0 \
		--tsv "$(DIST_DIR)/tech_pack_lexicon.tsv" \
		$(if $(wildcard $(DIST_DIR)/tech_lora.safetensors),--lora "$(DIST_DIR)/tech_lora.safetensors",)
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --norsrc --noextattr --keepParent "$(PACKS_BUILD_DIR)/classics-poetry.imepack" "$(RELEASE_DIR)/classics-poetry.imepack.zip"
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --norsrc --noextattr --keepParent "$(PACKS_BUILD_DIR)/tech-developer.imepack" "$(RELEASE_DIR)/tech-developer.imepack.zip"

define build-installer-pkg
@rm -rf "$(DIST_DIR)/pkg-root" "$(DIST_DIR)/component.plist" "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
@mkdir -p "$(DIST_DIR)/pkg-root/Library/Application Support/ChimeInstaller" "$(RELEASE_DIR)"
@COPYFILE_DISABLE=1 /usr/bin/ditto --norsrc --noextattr "$(APP_BUNDLE)" "$(DIST_DIR)/pkg-root/Library/Application Support/ChimeInstaller/Chime.app"
@chmod +x macos/Scripts/pkg_scripts/postinstall
@COPYFILE_DISABLE=1 /usr/bin/pkgbuild --analyze --root "$(DIST_DIR)/pkg-root" "$(DIST_DIR)/component.plist"
@/usr/bin/plutil -replace 0.BundleIsRelocatable -bool NO "$(DIST_DIR)/component.plist"
@/usr/bin/plutil -replace 0.BundleIsVersionChecked -bool NO "$(DIST_DIR)/component.plist"
@COPYFILE_DISABLE=1 /usr/bin/pkgbuild \
	--root "$(DIST_DIR)/pkg-root" \
	--component-plist "$(DIST_DIR)/component.plist" \
	--install-location "/" \
	--scripts macos/Scripts/pkg_scripts \
	--identifier "app.tyrchen.inputmethod.Chime.pkg" \
	--version "$(VERSION)" \
	$(1) \
	"$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
@rm -rf "$(DIST_DIR)/pkg-root" "$(DIST_DIR)/component.plist"
endef

.PHONY: macos-installer
macos-installer: macos-bundle
	$(call build-installer-pkg,$(if $(INSTALLER_SIGN_IDENTITY),--sign "$(INSTALLER_SIGN_IDENTITY)",))

release-artifacts: macos-installer build-public-packs
	@if [ -n "$(INSTALLER_SIGN_IDENTITY)" ]; then \
		pkgutil --check-signature "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"; \
	fi
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --norsrc --noextattr --keepParent "$(APP_BUNDLE)" "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@cp "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg" "$(RELEASE_DIR)/Chime-macos-arm64.pkg"
	@cp "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip" "$(RELEASE_DIR)/Chime-macos-arm64.zip"
	@cd "$(RELEASE_DIR)" && shasum -a 256 \
		"Chime-v$(VERSION)-macos-arm64.pkg" \
		"Chime-v$(VERSION)-macos-arm64.zip" \
		"Chime-macos-arm64.pkg" \
		"Chime-macos-arm64.zip" \
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
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --norsrc --noextattr --keepParent "$(APP_BUNDLE)" "$(DIST_DIR)/Chime-notarize-app.zip"
	$(call run-notarytool,$(DIST_DIR)/Chime-notarize-app.zip)
	@rm -f "$(DIST_DIR)/Chime-notarize-app.zip"
	@echo "2/4 Stapling notarization ticket to Chime.app and packaging .zip..."
	@xcrun stapler staple "$(APP_BUNDLE)"
	@xcrun stapler validate "$(APP_BUNDLE)"
	@spctl --assess --type exec --verbose=4 "$(APP_BUNDLE)"
	@rm -f "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@COPYFILE_DISABLE=1 /usr/bin/ditto -c -k --norsrc --noextattr --keepParent "$(APP_BUNDLE)" "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip"
	@echo "3/4 Building non-relocatable Developer ID signed .pkg installer and submitting to Apple Notary Service..."
	$(call build-installer-pkg,--sign "$(INSTALLER_SIGN_IDENTITY)")
	@pkgutil --check-signature "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	$(call run-notarytool,$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg)
	@echo "4/4 Stapling notarization ticket to .pkg and running Gatekeeper assessment..."
	@xcrun stapler staple "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@xcrun stapler validate "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@spctl --assess --type install --verbose=4 "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg"
	@cp "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.pkg" "$(RELEASE_DIR)/Chime-macos-arm64.pkg"
	@cp "$(RELEASE_DIR)/Chime-v$(VERSION)-macos-arm64.zip" "$(RELEASE_DIR)/Chime-macos-arm64.zip"
	@cd "$(RELEASE_DIR)" && shasum -a 256 \
		"Chime-v$(VERSION)-macos-arm64.pkg" \
		"Chime-v$(VERSION)-macos-arm64.zip" \
		"Chime-macos-arm64.pkg" \
		"Chime-macos-arm64.zip" \
		"classics-poetry.imepack.zip" \
		"tech-developer.imepack.zip" > SHA256SUMS.txt
	@echo "Successfully signed, notarized, and stapled all Chime v$(VERSION) release artifacts!"
	@cat "$(RELEASE_DIR)/SHA256SUMS.txt"

install-macos: macos-bundle
	@mkdir -p "$(HOME)/Library/Input Methods"
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app.new" "$(HOME)/Library/Input Methods/Chime.app.old"
	@COPYFILE_DISABLE=1 /usr/bin/ditto --norsrc --noextattr "$(APP_BUNDLE)" "$(HOME)/Library/Input Methods/Chime.app.new"
	@if [ -d "$(HOME)/Library/Input Methods/Chime.app" ]; then mv "$(HOME)/Library/Input Methods/Chime.app" "$(HOME)/Library/Input Methods/Chime.app.old"; fi
	@mv "$(HOME)/Library/Input Methods/Chime.app.new" "$(HOME)/Library/Input Methods/Chime.app"
	@rm -rf "$(HOME)/Library/Input Methods/Chime.app.old"
	@-"$(LSREGISTER)" -f -R -trusted "$(HOME)/Library/Input Methods/Chime.app"
	@-pkill -x Chime 2>/dev/null || true
	@sleep 0.4
	@/usr/bin/open -g -a "$(HOME)/Library/Input Methods/Chime.app"
	@echo "Installed and restarted Chime 拼音 v$(VERSION) at ~/Library/Input Methods/Chime.app"

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
