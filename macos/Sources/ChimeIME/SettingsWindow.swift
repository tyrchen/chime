import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Native macOS 15 SwiftUI Preferences Window controller for Chime IME.
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
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 560),
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
        panel.title = "导入第三方词库 (.scel / .bdict / .dict.yaml / .tsv) 或个人模型包 (.imepack)"
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
    @State private var config: ChimeConfigDTO = ChimeEngineBridge.shared.getConfig() ?? ChimeConfigDTO(
        scheme: "full",
        fuzzy: ChimeFuzzyDTO(
            z_zh: true, c_ch: true, s_sh: true, n_l: false, f_h: false, r_l: false,
            an_ang: true, en_eng: true, in_ing: true, ian_iang: false, uan_uang: false,
            fuzzy_penalty: 0.35
        ),
        pangu_spacing: true,
        chinese_punctuation: true,
        record_training_samples: false,
        anti_jitter_ms: 60,
        page_size: 5,
        candidate_layout: "horizontal"
    )
    @State private var status: ChimeStatusDTO? = ChimeEngineBridge.shared.getStatus()

    var body: some View {
        Form {
            Section("基础输入方案") {
                Picker("拼音方案", selection: $config.scheme) {
                    Text("全拼（支持超级简拼）").tag("full")
                    Text("小鹤双拼").tag("xiaohe")
                    Text("自然码双拼").tag("ziranma")
                    Text("微软双拼").tag("microsoft")
                }
                Picker("候选栏布局", selection: $config.candidate_layout) {
                    Text("横排单行").tag("horizontal")
                    Text("竖排列表").tag("vertical")
                }
                Stepper("每页候选个数：\(config.page_size)", value: $config.page_size, in: 3...9)
                Toggle("中英文与数字边界自动空格（代码编辑器自动豁免）", isOn: $config.pangu_spacing)
                Toggle("中文全角智能标点（成对引号与小数点智能识别）", isOn: $config.chinese_punctuation)
            }

            Section("模糊音（零词库体积膨胀）") {
                Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
                    GridRow {
                        Toggle("z ↔ zh", isOn: $config.fuzzy.z_zh)
                        Toggle("c ↔ ch", isOn: $config.fuzzy.c_ch)
                        Toggle("s ↔ sh", isOn: $config.fuzzy.s_sh)
                    }
                    GridRow {
                        Toggle("an ↔ ang", isOn: $config.fuzzy.an_ang)
                        Toggle("en ↔ eng", isOn: $config.fuzzy.en_eng)
                        Toggle("in ↔ ing", isOn: $config.fuzzy.in_ing)
                    }
                    GridRow {
                        Toggle("n ↔ l", isOn: $config.fuzzy.n_l)
                        Toggle("f ↔ h", isOn: $config.fuzzy.f_h)
                        Toggle("ian ↔ iang", isOn: $config.fuzzy.ian_iang)
                    }
                    GridRow {
                        Toggle("uan ↔ uang", isOn: $config.fuzzy.uan_uang)
                        Toggle("r ↔ l", isOn: $config.fuzzy.r_l)
                        Spacer()
                    }
                }
            }

            Section("个性化与词库引擎状态") {
                Toggle("记录本地打字训练对（仅存本机 SQLite，供导出 Colab 微调）", isOn: $config.record_training_samples)

                if let status {
                    LabeledContent("已挂载词条总数", value: "\(status.word_entries)")
                    LabeledContent("拼音索引组数", value: "\(status.pinyin_keys)")
                    LabeledContent("已学习个人习惯", value: "\(status.learned_habits)")
                    LabeledContent(
                        "Stage-2 端侧神经重排",
                        value: (status.neural_loaded || CoreMLStatefulReranker.shared.isLoaded) ? "MLX Metal / ANE 已就绪" : "统计双轨模式"
                    )
                }

                HStack {
                    Button("导入词库或 .imepack 模型包…") {
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
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 540)
        .onChange(of: config) { _, newValue in
            ChimeEngineBridge.shared.setConfig(newValue)
            status = ChimeEngineBridge.shared.getStatus()
        }
    }
}
