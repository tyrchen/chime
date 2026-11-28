import AppKit
import AVFoundation
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
            contentRect: NSRect(x: 0, y: 0, width: 680, height: 740),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        win.title = "Chime 输入法偏好设置"
        win.contentViewController = hosting
        win.minSize = NSSize(width: 660, height: 700)
        win.setContentSize(NSSize(width: 680, height: 740))
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

    public func presentTier2ModelFolderPanel() -> String? {
        let panel = NSOpenPanel()
        panel.title = "选择本地 MLX 4-bit Qwen3-Base 模型目录（含 config.json / vocab.json / merges.txt）"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            return url.path
        }
        return nil
    }
}

private enum SettingsTab: String, CaseIterable, Identifiable {
    case general = "常规与按键"
    case appearance = "排版与外观"
    case fuzzy = "模糊音"
    case lexicon = "词库与 AI"
    case voice = "语音输入"

    var id: String { rawValue }
}

struct ChimeSettingsView: View {
    @State private var selectedTab: SettingsTab = .general
    @State private var config: ChimeConfigDTO = ChimeEngineBridge.shared.getConfig() ?? .default
    @State private var status: ChimeStatusDTO? = ChimeEngineBridge.shared.getStatus()
    @State private var confirmClearHabits = false
    @ObservedObject private var updater = ChimeUpdateManager.shared
    @ObservedObject private var voiceModels = VoiceModelManager.shared
    @State private var confirmRemoveVoiceModel = false
    @State private var voiceApiKey = ""
    @State private var voiceProbe: VoiceEndpointProbe?
    @State private var voiceProbing = false
    @State private var micStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .audio)

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $selectedTab) {
                ForEach(SettingsTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 520)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            Divider()

            Group {
                switch selectedTab {
                case .general:
                    generalAndShortcutsTab
                case .appearance:
                    typographyAndAppearanceTab
                case .fuzzy:
                    fuzzyPinyinTab
                case .lexicon:
                    lexiconAndModelTab
                case .voice:
                    voiceInputTab
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 660, idealWidth: 680, minHeight: 700, idealHeight: 740)
        .onAppear {
            status = ChimeEngineBridge.shared.getStatus()
            micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
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
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                    GridRow {
                        Toggle("减号 / 等号（- / =）翻页", isOn: $config.minus_equal_paging)
                        Toggle("方括号（[ / ]）翻页", isOn: $config.bracket_paging)
                    }
                    GridRow {
                        Toggle("逗号 / 句号（, / .）翻页（多页时）", isOn: $config.comma_period_paging)
                        Toggle("分号 / 单引号（; / '）选第 2 / 3 词", isOn: $config.semicolon_apostrophe_select)
                    }
                }
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

            Section("表情与颜文字联想（100% 离线 · 绝不抢占首选）") {
                Toggle("候选栏联想 Emoji 表情（如：笑哭 → 😂、赞 → 👍，输入 biaoqing 浏览常用表情）", isOn: $config.emoji_suggestions)
                Toggle("候选栏联想颜文字（如：开心 → ヽ(✿ﾟ▽ﾟ)ノ，输入 yanwenzi 浏览全部）", isOn: $config.kaomoji_suggestions)
                Toggle("在终端与 IDE 中静默自动联想（手动输入 biaoqing / yanwenzi 仍可呼出）", isOn: $config.suppress_emoji_in_ide)
                    .disabled(!config.emoji_suggestions && !config.kaomoji_suggestions)
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
                            ? "Qwen3 BI-18L 混合精度 MLX Metal 已就绪（激活 \(status.active_loras ?? 0) 个 LoRA）"
                            : "统计双轨模式"
                    )
                    LabeledContent(
                        "高精度重排器 (Two-Tier)",
                        value: (status.tier2_loaded ?? false)
                            ? "已挂载本地 4-bit MLX 重排器"
                            : (config.tier2_enabled ? "已开启（等待有效模型目录）" : "未启用")
                    )
                }
            }

            Section("高精度本地两阶段重排（Two-Tier · 可选 · 100% 离线）") {
                Toggle(
                    "启用置信度门控 Two-Tier 本地重排（低置信度长句自动精排）",
                    isOn: $config.tier2_enabled
                )

                LabeledContent("本地模型目录") {
                    Text(
                        config.tier2_model_dir.isEmpty
                            ? ((status?.tier2_model_dir?.isEmpty == false)
                                ? (status?.tier2_model_dir ?? "")
                                : "默认：~/Library/Application Support/Chime/models/tier2")
                            : config.tier2_model_dir
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                }

                HStack(spacing: 10) {
                    Button("选择本地模型目录…") {
                        if let chosen = ChimeSettingsWindowController.shared.presentTier2ModelFolderPanel() {
                            config.tier2_model_dir = chosen
                            config.tier2_enabled = true
                        }
                    }
                    if !config.tier2_model_dir.isEmpty {
                        Button("恢复默认目录") {
                            config.tier2_model_dir = ""
                        }
                    }
                    Spacer()
                }

                Picker("置信度触发阈值 (margin)", selection: $config.tier2_margin_threshold) {
                    Text("保守 (1.5 · 仅极低置信度)").tag(1.5)
                    Text("平衡 (3.0 · 推荐)").tag(3.0)
                    Text("积极 (4.5 · 更多长句精排)").tag(4.5)
                }

                Picker("Tier-2 语义融合权重 (alpha)", selection: $config.tier2_alpha) {
                    Text("温和 (0.50)").tag(0.50)
                    Text("均衡 (0.65 · 推荐)").tag(0.65)
                    Text("偏重 Tier-2 (0.85)").tag(0.85)
                }

                Text("16 GB+ 统一内存推荐 Qwen3-4B-Base 4-bit (~2.3 GB)；8–16 GB 推荐 Qwen3-1.7B-Base 4-bit (~1.0 GB)。仅在 Tier-1 前两名分差低于阈值时触发，并自动保护「的/地/得」语法精度。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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

    private var micStatusText: String {
        switch micStatus {
        case .authorized: "已允许 Chime 使用麦克风"
        case .notDetermined: "尚未授权（首次按住快捷键时系统会询问）"
        case .denied, .restricted: "已拒绝：请在系统设置中允许 Chime 使用麦克风"
        @unknown default: "未知"
        }
    }

    private var voiceInputTab: some View {
        Form {
            Section("按住说话") {
                Toggle("启用语音输入", isOn: $config.voice.enabled)
                Picker("按住快捷键", selection: $config.voice.hotkey) {
                    Text("右 Option ⌥（推荐）").tag("right_option")
                    Text("右 Command ⌘").tag("right_command")
                    Text("右 Control ⌃").tag("right_control")
                    Text("Fn / 🌐").tag("fn")
                }
                Toggle("边说边出字", isOn: $config.voice.live_preview)
                Toggle("松键后智能补标点、断句", isOn: $config.voice.auto_punctuation)
                Toggle("整理口述：删去重复表达和口头填充词", isOn: $config.voice.tidy_dictation)
                Toggle("词库纠错建议（按 Tab 采用，不自动替换）", isOn: $config.voice.lexicon_suggestions)
                Text("只处理本次口述的最终稿，不改已输入的文字。关闭整理可保留原话。")
                    .font(.caption).foregroundStyle(.secondary)
                Text("按住 \(ChimeVoiceController.hotkeyLabel(config.voice.hotkey)) 约 0.3 秒开始录音，识别结果实时显示在光标处；松开后最终结果上屏，按 Esc 取消。密码框中不会启用录音。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if config.voice.hotkey == "fn" {
                    Text("使用 Fn 前，请在「系统设置 › 键盘」中把「按下 🌐 键时」设为「不执行任何操作」，避免与系统听写或表情面板冲突。")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }

            Section("麦克风权限") {
                HStack {
                    Text(micStatusText)
                    Spacer()
                    if micStatus == .notDetermined {
                        Button("请求权限") {
                            Task {
                                _ = await AVCaptureDevice.requestAccess(for: .audio)
                                micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                            }
                        }
                    } else if micStatus != .authorized {
                        Button("打开系统设置") {
                            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                                NSWorkspace.shared.open(url)
                            }
                        }
                    }
                }
            }

            Section("识别方式") {
                Picker("模型来源", selection: $config.voice.backend) {
                    Text("Streaming Paraformer（按需下载，离线识别）").tag("paraformer")
                    Text("自定义 HTTP 服务").tag("http")
                }
                if config.voice.backend != "http" {
                    HStack {
                        if let progress = voiceModels.progress {
                            ProgressView().controlSize(.small)
                            Text(progress)
                            Spacer()
                            Button("取消下载") { voiceModels.cancel() }
                        } else if voiceModels.installed {
                            Text("中英双语 INT8 · 已安装，可离线使用")
                            Spacer()
                            Button("移除模型…") { confirmRemoveVoiceModel = true }
                        } else {
                            Text("中英双语 INT8 · 未安装")
                            Spacer()
                            Button("下载模型（约 226 MiB）") { voiceModels.download() }
                        }
                    }
                    .onAppear { voiceModels.refresh() }
                    .confirmationDialog("移除本地 Paraformer 模型？", isPresented: $confirmRemoveVoiceModel) {
                        Button("移除", role: .destructive) { voiceModels.remove() }
                    } message: { Text("不会删除词库、标点模型或 HTTP 设置。以后可以重新下载。") }
                    if let error = voiceModels.error { Text(error).foregroundStyle(.red) }
                    Text("仅点击下载时从 Hugging Face 获取模型文件；校验后保存在本机。听写不会联网。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if config.voice.backend == "http" { voiceHTTPSettings }
            Section("录音") {
                Stepper("单次最长录音：\(config.voice.max_record_secs) 秒", value: $config.voice.max_record_secs, in: 10...600, step: 10)
            }
        }
        .formStyle(.grouped)
    }

    private var voiceHTTPSettings: some View {
        Group {
            Section("识别服务（本机或内网私有化部署的语音识别模型）") {
                TextField("服务地址", text: $config.voice.endpoint, prompt: Text("http://127.0.0.1:8000/v1/audio/transcriptions"))
                TextField("模型名称", text: $config.voice.model, prompt: Text("whisper-1"))
                Picker("识别语言", selection: $config.voice.language) {
                    Text("中文普通话 (zh)").tag("zh")
                    Text("粤语 (yue)").tag("yue")
                    Text("英文 (en)").tag("en")
                    Text("日文 (ja)").tag("ja")
                    Text("自动检测").tag("")
                }
                TextField("提示词（可选）", text: $config.voice.prompt, prompt: Text("用于偏向简体中文、标点或专业词汇"), axis: .vertical)
                    .lineLimit(2...4)
                SecureField("API Key（可选，保存在系统钥匙串）", text: Binding(
                    get: { voiceApiKey }, set: { voiceApiKey = $0; ChimeVoiceKeychain.write($0) }
                ))
                .onAppear { voiceApiKey = ChimeVoiceKeychain.read() ?? "" }
                HStack(spacing: 10) {
                    Button("测试连接") {
                        voiceProbing = true
                        voiceProbe = nil
                        Task {
                            voiceProbe = await ChimeVoiceController.probeEndpoint(config: config, apiKey: voiceApiKey)
                            voiceProbing = false
                        }
                    }
                    .disabled(voiceProbing)
                    if voiceProbing {
                        ProgressView()
                            .controlSize(.small)
                    }
                    if let probe = voiceProbe {
                        if probe.ok {
                            Text("✓ 连接成功（\(probe.latency_ms) ms）")
                                .foregroundStyle(.green)
                        } else {
                            Text("✗ \(probe.error ?? "失败")")
                                .foregroundStyle(.red)
                                .lineLimit(3)
                        }
                    }
                    Spacer()
                }
                Text("兼容 OpenAI 语音转写接口（POST multipart/form-data：file、model、language、prompt，返回 {\"text\": …}）。例如：whisper.cpp server（…/inference）、speaches / faster-whisper-server、vLLM、LocalAI（…/v1/audio/transcriptions）。音频只发送到这里填写的地址。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("高级") {
                Stepper("实时预览间隔：\(config.voice.partial_interval_ms) ms", value: $config.voice.partial_interval_ms, in: 200...5000, step: 100)
                Stepper("单次请求超时：\(config.voice.request_timeout_ms / 1000) 秒", value: $config.voice.request_timeout_ms, in: 2000...120_000, step: 1000)
            }
        }
    }
}
