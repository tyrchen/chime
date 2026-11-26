import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Native macOS SwiftUI 4-Tab Preferences Window controller for Chime IME (aligned with Sogou &
/// Doubao usability conventions, with integrated zero-telemetry updater and `.imepack` store).
@MainActor
public final class ChimeSettingsWindowController {
    public static let shared = ChimeSettingsWindowController()

    private var window: NSWindow?

    private init() {}

    public func showWindow() {
        if let existing = window {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let hosting = NSHostingController(rootView: ChimeSettingsView())
        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 560),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        win.title = "Chime 输入法偏好设置"
        win.contentViewController = hosting
        win.center()
        win.isReleasedWhenClosed = false
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = win
    }

    public func presentImportOpenPanel() {
        let panel = NSOpenPanel()
        panel.title = "导入词库 (.scel / .bdict / .dict.yaml / .tsv) 或扩展卡带 (.imepack / .imepack.zip)"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            _ = ChimeEngineBridge.shared.importAsset(url: url)
            CoreMLStatefulReranker.shared.loadInstalledModelIfPresent()
        }
    }
}

struct ChimeSettingsView: View {
    @State private var config: ChimeConfigDTO = ChimeEngineBridge.shared.getConfig() ?? .default
    @State private var status: ChimeStatusDTO? = ChimeEngineBridge.shared.getStatus()
    @State private var confirmClearHabits = false
    @ObservedObject private var updater = ChimeUpdateManager.shared

    var body: some View {
        TabView {
            generalAndShortcutsTab
                .tabItem {
                    Label("常规与按键", systemImage: "keyboard")
                }

            typographyAndAppearanceTab
                .tabItem {
                    Label("排版与外观", systemImage: "textformat.size")
                }

            fuzzyPinyinTab
                .tabItem {
                    Label("模糊音", systemImage: "waveform")
                }

            lexiconAndModelTab
                .tabItem {
                    Label("词库与 AI", systemImage: "cpu")
                }
        }
        .frame(minWidth: 590, minHeight: 540)
        .onAppear {
            status = ChimeEngineBridge.shared.getStatus()
            updater.ensureManifestLoadedQuietly()
        }
        .onChange(of: config) { _, newValue in
            ChimeEngineBridge.shared.setConfig(newValue)
            status = ChimeEngineBridge.shared.getStatus()
        }
        .alert("确认清空已学习的个人习惯词频？", isPresented: $confirmClearHabits) {
            Button("清空", role: .destructive) {
                ChimeEngineBridge.shared.clearUserHabits()
                status = ChimeEngineBridge.shared.getStatus()
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("这将清除本地 SQLite 中记录的个人选词偏好与退格纠偏记录，静态核心词库与扩展模型包不受影响。")
        }
    }

    private var generalAndShortcutsTab: some View {
        Form {
            Section("输入方案") {
                Picker("拼音方案", selection: $config.scheme) {
                    Text("全拼（支持超级简拼与自动纠错）").tag("full")
                    Text("小鹤双拼").tag("xiaohe")
                    Text("自然码双拼").tag("ziranma")
                    Text("微软双拼").tag("microsoft")
                }
            }

            Section("中英文切换习惯（对齐搜狗 / 豆包输入法）") {
                Picker("中英切换快捷键", selection: $config.english_toggle_key) {
                    Text("Shift（左 / 右 Shift 均可，默认推荐）").tag("both_shift")
                    Text("仅左 Shift").tag("left_shift")
                    Text("仅右 Shift").tag("right_shift")
                    Text("Caps Lock 大写锁定键").tag("caps_lock")
                    Text("关闭内部快捷键（仅使用系统切换键）").tag("none")
                }
                Picker("正在输入拼音时按 Shift", selection: $config.shift_commit_mode) {
                    Text("上屏已输入的英文字母，并切换至英文模式").tag("commit_and_toggle")
                    Text("仅上屏英文字母，保持中文输入状态").tag("commit_keep_chinese")
                }
                Toggle("中文模式下大写字母开头直接输入英文（如 Hello、MacBook）", isOn: $config.uppercase_triggers_english)
                Toggle("英文拼写自动纠错（支持中文混输与 En 英文模式，按退格键 ⌫ 一键撤回）", isOn: $config.english_auto_correct)
                Toggle("切换中英文或全半角标点时在光标旁显示状态浮窗（中 / En）", isOn: $config.show_status_toast)
            }

            Section("翻页与快速选词热键") {
                Toggle("使用减号 / 等号（- / =）翻页", isOn: $config.minus_equal_paging)
                Toggle("使用左 / 右方括号（[ / ]）翻页", isOn: $config.bracket_paging)
                Toggle("使用逗号 / 句号（, / .）翻页（多页候选时生效）", isOn: $config.comma_period_paging)
                Toggle("使用分号 / 单引号（; / '）快速选择第 2 / 第 3 候选词", isOn: $config.semicolon_apostrophe_select)
            }

            Section("版本与自动更新（零遥测 · SHA-256 + Apple 公证校验）") {
                HStack {
                    Text("当前版本：v\(updater.currentVersion)")
                        .fontWeight(.medium)
                    Spacer()
                    if updater.isChecking || updater.isUpdatingApp {
                        ProgressView()
                            .controlSize(.small)
                    }
                    if let upd = updater.availableAppUpdate {
                        Button("立即升级至 v\(upd.version)") {
                            updater.installAvailableAppUpdate()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(updater.isUpdatingApp)
                    } else {
                        Button("检查更新") {
                            updater.checkForUpdatesInteractive(openSettingsWindow: false)
                        }
                        .disabled(updater.isChecking || updater.isUpdatingApp)
                    }
                }
                Toggle("自动检查更新（每 24 小时仅只读拉取静态版本号，零设备指纹与按键上传）", isOn: $config.auto_check_updates)
                if !updater.statusMessage.isEmpty {
                    Text(updater.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var typographyAndAppearanceTab: some View {
        Form {
            Section("智能排版与标点") {
                Toggle("中英文与数字前后自动插入半角空格（如：我有 100 个、使用 Rust 开发）", isOn: $config.pangu_spacing)
                Toggle("中文状态下使用全角标点（自动豁免 3.14 小数点与 12:30 时间冒号，Ctrl+. 快速切换）", isOn: $config.chinese_punctuation)
            }

            Section("候选栏外观") {
                Picker("候选栏排列方向", selection: $config.candidate_layout) {
                    Text("横排单行（按下键 ↓ 可展开 5×5 面板）").tag("horizontal")
                    Text("竖排列表").tag("vertical")
                }
                Stepper("每页候选词数量：\(config.page_size) 个", value: $config.page_size, in: 3...9)
                Picker("候选词字体大小", selection: $config.candidate_font_size) {
                    Text("紧凑 (14 pt)").tag(14)
                    Text("标准 (15 pt，默认)").tag(15)
                    Text("舒适大号 (17 pt)").tag(17)
                    Text("特大字号 (19 pt)").tag(19)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var fuzzyPinyinTab: some View {
        Form {
            Section("快捷预设") {
                HStack(spacing: 12) {
                    Button("开启全部常用模糊音") {
                        config.fuzzy.z_zh = true
                        config.fuzzy.c_ch = true
                        config.fuzzy.s_sh = true
                        config.fuzzy.an_ang = true
                        config.fuzzy.en_eng = true
                        config.fuzzy.in_ing = true
                        config.fuzzy.ian_iang = true
                        config.fuzzy.uan_uang = true
                    }
                    Button("全部关闭（严格标准音）") {
                        config.fuzzy.z_zh = false
                        config.fuzzy.c_ch = false
                        config.fuzzy.s_sh = false
                        config.fuzzy.n_l = false
                        config.fuzzy.f_h = false
                        config.fuzzy.r_l = false
                        config.fuzzy.an_ang = false
                        config.fuzzy.en_eng = false
                        config.fuzzy.in_ing = false
                        config.fuzzy.ian_iang = false
                        config.fuzzy.uan_uang = false
                    }
                    Spacer()
                }
            }

            Section("声母与韵母模糊对（零词库膨胀实时代数展开）") {
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 10) {
                    GridRow {
                        Toggle("z ↔ zh（平翘舌）", isOn: $config.fuzzy.z_zh)
                        Toggle("c ↔ ch（平翘舌）", isOn: $config.fuzzy.c_ch)
                        Toggle("s ↔ sh（平翘舌）", isOn: $config.fuzzy.s_sh)
                    }
                    GridRow {
                        Toggle("an ↔ ang（前后鼻音）", isOn: $config.fuzzy.an_ang)
                        Toggle("en ↔ eng（前后鼻音）", isOn: $config.fuzzy.en_eng)
                        Toggle("in ↔ ing（前后鼻音）", isOn: $config.fuzzy.in_ing)
                    }
                    GridRow {
                        Toggle("ian ↔ iang", isOn: $config.fuzzy.ian_iang)
                        Toggle("uan ↔ uang", isOn: $config.fuzzy.uan_uang)
                        Toggle("n ↔ l（鼻边音）", isOn: $config.fuzzy.n_l)
                    }
                    GridRow {
                        Toggle("f ↔ h", isOn: $config.fuzzy.f_h)
                        Toggle("r ↔ l", isOn: $config.fuzzy.r_l)
                        Spacer()
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var lexiconAndModelTab: some View {
        Form {
            Section("端侧引擎与神经网络状态") {
                if let status {
                    LabeledContent("已挂载词条总数", value: "\(status.word_entries)")
                    LabeledContent("拼音索引组数", value: "\(status.pinyin_keys)")
                    LabeledContent("已挂载扩展卡带 (.imepack)", value: "\(status.installed_packs ?? 0) 个")
                    LabeledContent("已学习个人习惯词条", value: "\(status.learned_habits) 条")
                    LabeledContent(
                        "端侧神经解码引擎",
                        value: (status.neural_loaded || CoreMLStatefulReranker.shared.isLoaded)
                            ? "Qwen3 28L 4-bit MLX Metal 已就绪（激活 \(status.active_loras ?? 0) 个 LoRA）"
                            : "统计双轨模式"
                    )
                }
            }

            Section("官方扩展卡带一键安装与更新（SHA-256 校验 · 热挂载免重启）") {
                ForEach(OfficialPackCatalogItem.catalog) { item in
                    let installed = status?.pack_details?.first(where: { $0.id == item.id })
                    let remoteVer = updater.latestManifest?.version
                    let hasPackUpdate = if let inst = installed, let rVer = remoteVer {
                        ChimeUpdateManager.isVersionNewer(rVer, than: inst.version)
                    } else {
                        false
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .center, spacing: 8) {
                            Text(item.title)
                                .fontWeight(.semibold)
                            Text(item.routingBadge)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.12))
                                .clipShape(Capsule())
                            Spacer()
                            if updater.activePackDownloadId == item.id {
                                ProgressView()
                                    .controlSize(.small)
                                Text("下载中…")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else if let inst = installed {
                                if hasPackUpdate, let rVer = remoteVer {
                                    Button("更新至 v\(rVer)") {
                                        updater.installOrUpdateOfficialPack(packId: item.id) {
                                            status = ChimeEngineBridge.shared.getStatus()
                                        }
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.small)
                                } else {
                                    Text("已挂载 v\(inst.version)")
                                        .font(.caption)
                                        .foregroundStyle(.green)
                                }
                                Button("卸载") {
                                    _ = ChimeEngineBridge.shared.removePack(id: item.id)
                                    status = ChimeEngineBridge.shared.getStatus()
                                }
                                .controlSize(.small)
                            } else {
                                Button("一键安装 (\(item.approxSize))") {
                                    updater.installOrUpdateOfficialPack(packId: item.id) {
                                        status = ChimeEngineBridge.shared.getStatus()
                                    }
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                            }
                        }
                        Text(item.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 3)
                }
                if !updater.statusMessage.isEmpty {
                    Text(updater.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("本地词库导入与个人数据管理") {
                Toggle("记录本地打字训练样本（仅存本机 SQLite，供导出 Colab 微调个人 LoRA）", isOn: $config.record_training_samples)

                HStack(spacing: 10) {
                    Button("导入词库或 .imepack…") {
                        ChimeSettingsWindowController.shared.presentImportOpenPanel()
                        status = ChimeEngineBridge.shared.getStatus()
                    }
                    Button("热重载配置与模型") {
                        ChimeEngineBridge.shared.reloadAssets()
                        CoreMLStatefulReranker.shared.loadInstalledModelIfPresent()
                        if let fresh = ChimeEngineBridge.shared.getConfig() {
                            config = fresh
                        }
                        status = ChimeEngineBridge.shared.getStatus()
                    }
                    Spacer()
                    Button("清空个人习惯词频…", role: .destructive) {
                        confirmClearHabits = true
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}
