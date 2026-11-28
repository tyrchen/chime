import AppKit
import Carbon
import CChimeFFI
import Foundation
import InputMethodKit

/// Decoded candidate item returned across the narrow C-ABI boundary.
public struct ChimeCandidate: Decodable, Sendable, Equatable {
    public let text: String
    public let score: Float
    public let kind: String
    public let consumed_bytes: Int
}

/// Synchronous keystroke response snapshot (< 2ms hot path).
public struct ChimeResponse: Decodable, Sendable, Equatable {
    public let handled: Bool
    public let committed_text: String
    public let preedit_text: String
    public let candidates: [ChimeCandidate]
    public let highlighted_index: Int
    public let page_index: Int
    public let total_pages: Int
    public let inline_english_mode: Bool
    public let chinese_punctuation: Bool
    public let grid_expanded: Bool
    public let ghost_text: String
    public let candidate_layout: String
    public let candidate_font_size: Int
    public let show_status_toast: Bool
    public let replace_prev_chars: Int
    public let toast_override: String
    public let generation_id: UInt64

    enum CodingKeys: String, CodingKey {
        case handled
        case committed_text
        case preedit_text
        case candidates
        case highlighted_index
        case page_index
        case total_pages
        case inline_english_mode
        case chinese_punctuation
        case grid_expanded
        case ghost_text
        case candidate_layout
        case candidate_font_size
        case show_status_toast
        case replace_prev_chars
        case toast_override
        case generation_id
    }

    public init(
        handled: Bool,
        committed_text: String,
        preedit_text: String,
        candidates: [ChimeCandidate],
        highlighted_index: Int,
        page_index: Int,
        total_pages: Int,
        inline_english_mode: Bool,
        chinese_punctuation: Bool,
        grid_expanded: Bool,
        ghost_text: String,
        candidate_layout: String,
        candidate_font_size: Int = 15,
        show_status_toast: Bool = true,
        replace_prev_chars: Int = 0,
        toast_override: String = "",
        generation_id: UInt64
    ) {
        self.handled = handled
        self.committed_text = committed_text
        self.preedit_text = preedit_text
        self.candidates = candidates
        self.highlighted_index = highlighted_index
        self.page_index = page_index
        self.total_pages = total_pages
        self.inline_english_mode = inline_english_mode
        self.chinese_punctuation = chinese_punctuation
        self.grid_expanded = grid_expanded
        self.ghost_text = ghost_text
        self.candidate_layout = candidate_layout
        self.candidate_font_size = candidate_font_size
        self.show_status_toast = show_status_toast
        self.replace_prev_chars = replace_prev_chars
        self.toast_override = toast_override
        self.generation_id = generation_id
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.handled = try c.decode(Bool.self, forKey: .handled)
        self.committed_text = try c.decode(String.self, forKey: .committed_text)
        self.preedit_text = try c.decode(String.self, forKey: .preedit_text)
        self.candidates = try c.decode([ChimeCandidate].self, forKey: .candidates)
        self.highlighted_index = try c.decode(Int.self, forKey: .highlighted_index)
        self.page_index = try c.decode(Int.self, forKey: .page_index)
        self.total_pages = try c.decode(Int.self, forKey: .total_pages)
        self.inline_english_mode = try c.decode(Bool.self, forKey: .inline_english_mode)
        self.chinese_punctuation = try c.decode(Bool.self, forKey: .chinese_punctuation)
        self.grid_expanded = try c.decode(Bool.self, forKey: .grid_expanded)
        self.ghost_text = try c.decode(String.self, forKey: .ghost_text)
        self.candidate_layout = try c.decode(String.self, forKey: .candidate_layout)
        self.candidate_font_size = (try? c.decode(Int.self, forKey: .candidate_font_size)) ?? 15
        self.show_status_toast = (try? c.decode(Bool.self, forKey: .show_status_toast)) ?? true
        self.replace_prev_chars = (try? c.decode(Int.self, forKey: .replace_prev_chars)) ?? 0
        self.toast_override = (try? c.decode(String.self, forKey: .toast_override)) ?? ""
        self.generation_id = try c.decode(UInt64.self, forKey: .generation_id)
    }

    public static let unhandled = ChimeResponse(
        handled: false,
        committed_text: "",
        preedit_text: "",
        candidates: [],
        highlighted_index: 0,
        page_index: 0,
        total_pages: 0,
        inline_english_mode: false,
        chinese_punctuation: true,
        grid_expanded: false,
        ghost_text: "",
        candidate_layout: "horizontal",
        candidate_font_size: 15,
        show_status_toast: true,
        generation_id: 0
    )
}

/// Fuzzy Pinyin configuration mirrored for SwiftUI Settings binding.
public struct ChimeFuzzyDTO: Codable, Sendable, Equatable {
    public var z_zh: Bool
    public var c_ch: Bool
    public var s_sh: Bool
    public var n_l: Bool
    public var f_h: Bool
    public var r_l: Bool
    public var an_ang: Bool
    public var en_eng: Bool
    public var in_ing: Bool
    public var ian_iang: Bool
    public var uan_uang: Bool
    public var fuzzy_penalty: Float

    public static let `default` = ChimeFuzzyDTO(
        z_zh: true, c_ch: true, s_sh: true, n_l: false, f_h: false, r_l: false,
        an_ang: true, en_eng: true, in_ing: true, ian_iang: true, uan_uang: true,
        fuzzy_penalty: 0.52
    )
}

/// Full engine configuration mirrored for SwiftUI Settings binding.
public struct ChimeConfigDTO: Codable, Sendable, Equatable {
    public var scheme: String
    public var fuzzy: ChimeFuzzyDTO
    public var pangu_spacing: Bool
    public var chinese_punctuation: Bool
    public var english_toggle_key: String
    public var shift_commit_mode: String
    public var uppercase_triggers_english: Bool
    public var english_auto_correct: Bool
    public var show_status_toast: Bool
    public var minus_equal_paging: Bool
    public var bracket_paging: Bool
    public var comma_period_paging: Bool
    public var semicolon_apostrophe_select: Bool
    public var record_training_samples: Bool
    public var anti_jitter_ms: UInt64
    public var page_size: Int
    public var candidate_font_size: Int
    public var candidate_layout: String
    public var auto_check_updates: Bool
    public var emoji_suggestions: Bool
    public var kaomoji_suggestions: Bool
    public var suppress_emoji_in_ide: Bool

    public static let `default` = ChimeConfigDTO(
        scheme: "full",
        fuzzy: .default,
        pangu_spacing: true,
        chinese_punctuation: true,
        english_toggle_key: "both_shift",
        shift_commit_mode: "commit_and_toggle",
        uppercase_triggers_english: true,
        english_auto_correct: true,
        show_status_toast: true,
        minus_equal_paging: true,
        bracket_paging: false,
        comma_period_paging: false,
        semicolon_apostrophe_select: false,
        record_training_samples: false,
        anti_jitter_ms: 60,
        page_size: 5,
        candidate_font_size: 15,
        candidate_layout: "horizontal",
        auto_check_updates: true,
        emoji_suggestions: true,
        kaomoji_suggestions: true,
        suppress_emoji_in_ide: false
    )

    public init(
        scheme: String,
        fuzzy: ChimeFuzzyDTO,
        pangu_spacing: Bool,
        chinese_punctuation: Bool,
        english_toggle_key: String,
        shift_commit_mode: String,
        uppercase_triggers_english: Bool,
        english_auto_correct: Bool,
        show_status_toast: Bool,
        minus_equal_paging: Bool,
        bracket_paging: Bool,
        comma_period_paging: Bool,
        semicolon_apostrophe_select: Bool,
        record_training_samples: Bool,
        anti_jitter_ms: UInt64,
        page_size: Int,
        candidate_font_size: Int,
        candidate_layout: String,
        auto_check_updates: Bool,
        emoji_suggestions: Bool,
        kaomoji_suggestions: Bool,
        suppress_emoji_in_ide: Bool
    ) {
        self.scheme = scheme
        self.fuzzy = fuzzy
        self.pangu_spacing = pangu_spacing
        self.chinese_punctuation = chinese_punctuation
        self.english_toggle_key = english_toggle_key
        self.shift_commit_mode = shift_commit_mode
        self.uppercase_triggers_english = uppercase_triggers_english
        self.english_auto_correct = english_auto_correct
        self.show_status_toast = show_status_toast
        self.minus_equal_paging = minus_equal_paging
        self.bracket_paging = bracket_paging
        self.comma_period_paging = comma_period_paging
        self.semicolon_apostrophe_select = semicolon_apostrophe_select
        self.record_training_samples = record_training_samples
        self.anti_jitter_ms = anti_jitter_ms
        self.page_size = page_size
        self.candidate_font_size = candidate_font_size
        self.candidate_layout = candidate_layout
        self.auto_check_updates = auto_check_updates
        self.emoji_suggestions = emoji_suggestions
        self.kaomoji_suggestions = kaomoji_suggestions
        self.suppress_emoji_in_ide = suppress_emoji_in_ide
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.scheme = try c.decodeIfPresent(String.self, forKey: .scheme) ?? "full"
        self.fuzzy = try c.decodeIfPresent(ChimeFuzzyDTO.self, forKey: .fuzzy) ?? .default
        self.pangu_spacing = try c.decodeIfPresent(Bool.self, forKey: .pangu_spacing) ?? true
        self.chinese_punctuation = try c.decodeIfPresent(Bool.self, forKey: .chinese_punctuation) ?? true
        self.english_toggle_key = try c.decodeIfPresent(String.self, forKey: .english_toggle_key) ?? "both_shift"
        self.shift_commit_mode = try c.decodeIfPresent(String.self, forKey: .shift_commit_mode) ?? "commit_and_toggle"
        self.uppercase_triggers_english = try c.decodeIfPresent(Bool.self, forKey: .uppercase_triggers_english) ?? true
        self.english_auto_correct = try c.decodeIfPresent(Bool.self, forKey: .english_auto_correct) ?? true
        self.show_status_toast = try c.decodeIfPresent(Bool.self, forKey: .show_status_toast) ?? true
        self.minus_equal_paging = try c.decodeIfPresent(Bool.self, forKey: .minus_equal_paging) ?? true
        self.bracket_paging = try c.decodeIfPresent(Bool.self, forKey: .bracket_paging) ?? false
        self.comma_period_paging = try c.decodeIfPresent(Bool.self, forKey: .comma_period_paging) ?? false
        self.semicolon_apostrophe_select = try c.decodeIfPresent(Bool.self, forKey: .semicolon_apostrophe_select) ?? false
        self.record_training_samples = try c.decodeIfPresent(Bool.self, forKey: .record_training_samples) ?? false
        self.anti_jitter_ms = try c.decodeIfPresent(UInt64.self, forKey: .anti_jitter_ms) ?? 60
        self.page_size = try c.decodeIfPresent(Int.self, forKey: .page_size) ?? 5
        self.candidate_font_size = try c.decodeIfPresent(Int.self, forKey: .candidate_font_size) ?? 15
        self.candidate_layout = try c.decodeIfPresent(String.self, forKey: .candidate_layout) ?? "horizontal"
        self.auto_check_updates = try c.decodeIfPresent(Bool.self, forKey: .auto_check_updates) ?? true
        self.emoji_suggestions = try c.decodeIfPresent(Bool.self, forKey: .emoji_suggestions) ?? true
        self.kaomoji_suggestions = try c.decodeIfPresent(Bool.self, forKey: .kaomoji_suggestions) ?? true
        self.suppress_emoji_in_ide = try c.decodeIfPresent(Bool.self, forKey: .suppress_emoji_in_ide) ?? false
    }
}

/// Summary of a mounted `.imepack` cartridge reported by `chime_get_status_json`.
public struct ChimeInstalledPackDTO: Decodable, Sendable, Equatable, Identifiable {
    public let id: String
    public let name: String
    public let version: String
    public let has_lora: Bool
}

/// Live engine diagnostic metrics displayed in SwiftUI Settings.
public struct ChimeStatusDTO: Decodable, Sendable, Equatable {
    public let scheme: String
    public let pinyin_keys: Int
    public let word_entries: Int
    public let learned_habits: Int
    public let neural_loaded: Bool
    public let installed_packs: Int?
    public let active_loras: Int?
    public let pack_details: [ChimeInstalledPackDTO]?
}

/// MainActor-isolated coordinator owning the persistent Rust `ChimeEngineHandle` and lightweight
/// per-controller metadata keyed by `ObjectIdentifier`.
@MainActor
public final class ChimeEngineBridge {
    public static let shared = ChimeEngineBridge()

    private struct ControllerSlot {
        var sessionId: UInt64
        var lastCaretRect: NSRect
        var hasObservedCaret: Bool
        var hadMarkedPreedit: Bool
        weak var lastClientObject: AnyObject?
    }

    private var handle: OpaquePointer?
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private var slots: [ObjectIdentifier: ControllerSlot] = [:]
    private var pollTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
    public var activeControllerKey: ObjectIdentifier?

    // Engine-wide modifier tap & English mode state (shared across windows like Sogou / Doubao)
    private var cachedConfig: ChimeConfigDTO = .default
    private var inlineEnglishModeGlobal: Bool = false
    private var shiftDownUptime: TimeInterval?
    private var shiftDownSide: UInt16? // 0x38 = left, 0x3C = right, 0 = generic
    private var keyOrMouseDuringShift: Bool = false
    private var lastProcessedFlagsUptime: TimeInterval = 0
    private var lastProcessedFlagsMask: UInt = 0
    private var lastModeToggleUptime: TimeInterval = 0
    private var enStatusItem: NSStatusItem?

    private init() {
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("Chime", isDirectory: true)
            .path ?? ""
        let bundleRes = Bundle.main.resourcePath ?? ""

        self.handle = appSupport.withCString { supCStr in
            bundleRes.withCString { resCStr in
                chime_engine_new_with_dirs(supCStr, resCStr)
            }
        }
        if let cfg = self.getConfig() {
            self.cachedConfig = cfg
        }

        CandidatePanel.shared.onSelectCandidate = { [weak self] pageRelIdx in
            guard let self, let key = self.activeControllerKey else { return }
            let resp = self.selectCandidateFromPanel(controllerKey: key, index: pageRelIdx)
            self.applyResponseToStoredClient(controllerKey: key, resp: resp)
        }

        CandidatePanel.shared.onToggleGrid = { [weak self] in
            guard let self, let key = self.activeControllerKey else { return }
            let resp = self.toggleGrid(controllerKey: key)
            self.applyResponseToStoredClient(controllerKey: key, resp: resp)
        }

        CandidatePanel.shared.onTogglePunctuation = { [weak self] in
            guard let self, let key = self.activeControllerKey else { return }
            let resp = self.togglePunctuation(controllerKey: key)
            self.applyResponseToStoredClient(controllerKey: key, resp: resp)
            if resp.show_status_toast {
                let caret = self.liveCaretOrMouseRect(for: key)
                StatusToastPanel.shared.show(
                    badge: resp.chinese_punctuation ? "中。" : "英.",
                    caretRect: caret
                )
            }
        }

        CandidatePanel.shared.onToggleEnglish = { [weak self] in
            guard let self else { return }
            let resp = self.toggleEnglish(controllerKey: self.activeControllerKey)
            if let key = self.activeControllerKey {
                self.applyResponseToStoredClient(controllerKey: key, resp: resp)
            }
            self.presentModeChangeFeedback(resp: resp, controllerKey: self.activeControllerKey)
        }

        installGlobalModifierMonitor()
    }

    private func installGlobalModifierMonitor() {
        // `.flagsChanged` and mouse clicks require ZERO Accessibility permission on macOS and catch
        // bare `Shift` taps reliably even in apps/terminals that do not forward idle `flagsChanged`
        // over `IMKInputSession` IPC!
        NSEvent.addGlobalMonitorForEvents(matching: [.flagsChanged, .leftMouseDown, .rightMouseDown]) { [weak self] event in
            let type = event.type
            let input = ChimeKeyInput(event)
            Task { @MainActor [weak self] in
                guard let self else { return }
                if type == .leftMouseDown || type == .rightMouseDown {
                    self.keyOrMouseDuringShift = true
                    if self.inlineEnglishModeGlobal, let key = self.activeControllerKey {
                        self.reset(controllerKey: key)
                    }
                    return
                }
                guard let input else { return }
                guard Self.isChimeSelectedInTIS() else {
                    self.syncMenuBarEnBadge()
                    return
                }
                if let resp = self.pollModifierToggle(
                    controllerKey: self.activeControllerKey,
                    keyCode: input.keyCode,
                    flags: input.flags
                ) {
                    if let key = self.activeControllerKey {
                        self.applyResponseToStoredClient(controllerKey: key, resp: resp)
                    }
                    self.presentModeChangeFeedback(resp: resp, controllerKey: self.activeControllerKey)
                }
            }
        }

        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.syncMenuBarEnBadge()
            }
        }
    }

    public nonisolated static func isChimeSelectedInTIS() -> Bool {
        guard let src = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
              let ptr = TISGetInputSourceProperty(src, kTISPropertyInputSourceID)
        else {
            return false
        }
        let srcId = Unmanaged<CFString>.fromOpaque(ptr).takeUnretainedValue() as String
        return srcId.hasPrefix("app.tyrchen.inputmethod.Chime")
    }

    private func syncMenuBarEnBadge() {
        let shouldShow = inlineEnglishModeGlobal && Self.isChimeSelectedInTIS()
        if shouldShow {
            if enStatusItem == nil {
                let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
                if let btn = item.button {
                    btn.title = "En"
                    btn.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .semibold)
                    btn.toolTip = "Chime 当前处于英文输入模式（按 Shift 或点击切回中文）"
                    btn.target = self
                    btn.action = #selector(handleEnStatusBadgeClick(_:))
                }
                enStatusItem = item
            }
        } else if let item = enStatusItem {
            NSStatusBar.system.removeStatusItem(item)
            enStatusItem = nil
        }
    }

    @objc private func handleEnStatusBadgeClick(_ sender: Any?) {
        let resp = toggleEnglish(controllerKey: activeControllerKey)
        if let key = activeControllerKey {
            applyResponseToStoredClient(controllerKey: key, resp: resp)
        }
        presentModeChangeFeedback(resp: resp, controllerKey: activeControllerKey)
    }

    public func presentModeChangeFeedback(
        resp: ChimeResponse,
        controllerKey: ObjectIdentifier?,
        preferredRect: NSRect? = nil
    ) {
        syncMenuBarEnBadge()
        guard resp.show_status_toast else { return }
        let rect = preferredRect
            ?? CandidatePanel.shared.activeAnchorCaret
            ?? liveCaretOrMouseRect(for: controllerKey)
        StatusToastPanel.shared.show(
            badge: resp.inline_english_mode ? "En" : "中",
            caretRect: rect
        )
    }

    /// Queries a validated screen-space caret rectangle from an `IMKTextInput` client, preferring
    /// the start of the active marked preedit (so multi-line wrapping & Electron editors stay rock
    /// solid) and rejecting degenerate `(0, 0)` or out-of-bounds heights.
    public nonisolated static func queryValidatedCaretRect(from client: IMKTextInput) -> NSRect? {
        var indices: [Int] = []
        let marked = client.markedRange()
        if marked.location != NSNotFound && marked.length > 0 {
            indices.append(marked.location)
        }
        let sel = client.selectedRange()
        if sel.location != NSNotFound {
            indices.append(sel.location)
        }
        if !indices.contains(0) {
            indices.append(0)
        }

        for idx in indices {
            var lineRect = NSRect.zero
            client.attributes(forCharacterIndex: idx, lineHeightRectangle: &lineRect)
            let ok = MainActor.assumeIsolated {
                Self.isPlausibleCaretRect(lineRect)
            }
            if ok {
                return lineRect
            }
        }
        return nil
    }

    private static func isPlausibleCaretRect(_ rect: NSRect) -> Bool {
        guard rect.origin.x.isFinite && rect.origin.y.isFinite
            && rect.size.width.isFinite && rect.size.height.isFinite
        else {
            return false
        }
        if rect.origin.x == 0 && rect.origin.y == 0 {
            return false
        }
        if rect.size.height < 4.0 || rect.size.height > 120.0 {
            return false
        }
        return NSScreen.screens.contains(where: { screen in
            screen.frame.insetBy(dx: -40, dy: -40).contains(rect.origin)
        })
    }

    public func liveCaretOrMouseRect(for controllerKey: ObjectIdentifier?) -> NSRect {
        if let activeAnchor = CandidatePanel.shared.activeAnchorCaret {
            return activeAnchor
        }
        if let key = controllerKey ?? activeControllerKey, let slot = slots[key] {
            if let client = slot.lastClientObject as? IMKTextInput,
               let live = Self.queryValidatedCaretRect(from: client)
            {
                return live
            }
            if slot.hasObservedCaret && Self.isPlausibleCaretRect(slot.lastCaretRect) {
                return slot.lastCaretRect
            }
        }
        let mouse = NSEvent.mouseLocation
        return NSRect(x: mouse.x, y: mouse.y - 18, width: 1, height: 18)
    }

    public func isInlineEnglish(controllerKey: ObjectIdentifier?) -> Bool {
        inlineEnglishModeGlobal
    }

    public func caretRect(for controllerKey: ObjectIdentifier) -> NSRect {
        liveCaretOrMouseRect(for: controllerKey)
    }

    public func hadMarkedPreedit(for controllerKey: ObjectIdentifier) -> Bool {
        slots[controllerKey]?.hadMarkedPreedit ?? false
    }

    public func setHadMarkedPreedit(for controllerKey: ObjectIdentifier, value: Bool) {
        if var slot = slots[controllerKey] {
            slot.hadMarkedPreedit = value
            slots[controllerKey] = slot
        }
    }

    public func ensureSlot(controllerKey: ObjectIdentifier, client: AnyObject?) {
        activeControllerKey = controllerKey
        if var existing = slots[controllerKey] {
            if let client {
                existing.lastClientObject = client
            }
            slots[controllerKey] = existing
            return
        }
        let bundleId = (client as? IMKTextInput)?.bundleIdentifier() ?? ""
        activate(controllerKey: controllerKey, appBundleId: bundleId, client: client)
    }

    public func activate(controllerKey: ObjectIdentifier, appBundleId: String, client: AnyObject?) {
        activeControllerKey = controllerKey
        syncMenuBarEnBadge()
        if var existing = slots[controllerKey] {
            existing.lastClientObject = client
            slots[controllerKey] = existing
            return
        }
        guard let handle else { return }
        let sid = appBundleId.withCString { cStr in
            chime_session_create(handle, cStr)
        }
        let mouse = NSEvent.mouseLocation
        slots[controllerKey] = ControllerSlot(
            sessionId: sid,
            lastCaretRect: NSRect(x: mouse.x, y: mouse.y - 18, width: 1, height: 18),
            hasObservedCaret: false,
            hadMarkedPreedit: false,
            lastClientObject: client
        )
    }

    public func deactivate(controllerKey: ObjectIdentifier) {
        pollTasks.removeValue(forKey: controllerKey)?.cancel()
        if activeControllerKey == controllerKey {
            activeControllerKey = nil
        }
        CandidatePanel.shared.hide()
        StatusToastPanel.shared.hide()
        syncMenuBarEnBadge()
        guard let slot = slots.removeValue(forKey: controllerKey), let handle, slot.sessionId != 0 else {
            return
        }
        chime_session_destroy(handle, slot.sessionId)
    }

    public func reset(controllerKey: ObjectIdentifier) {
        CandidatePanel.shared.hide()
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else { return }
        chime_reset_session(handle, slot.sessionId)
    }

    public func commitComposition(controllerKey: ObjectIdentifier) {
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else { return }
        let rawPtr = chime_commit_composition(handle, slot.sessionId)
        let resp = decodeAndFree(rawPtr)
        if resp.candidates.isEmpty && resp.ghost_text.isEmpty {
            CandidatePanel.shared.hide()
        } else {
            CandidatePanel.shared.update(response: resp, caretRect: slot.lastCaretRect)
        }
    }

    public func syncSurroundingContext(controllerKey: ObjectIdentifier, leftContext: String) {
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else { return }
        leftContext.withCString { cStr in
            chime_sync_context(handle, slot.sessionId, cStr)
        }
    }

    public func registerClient(controllerKey: ObjectIdentifier, client: AnyObject) {
        if var slot = slots[controllerKey] {
            slot.lastClientObject = client
            slots[controllerKey] = slot
        }
    }

    /// Detects clean modifier taps (`both_shift`, `left_shift`, `right_shift`, `caps_lock`) across
    /// both `IMKInputController` and the global `.flagsChanged` monitor, deduplicating dual-delivered
    /// events and handling Carbon IPC events where `keyCode == 0` via NX device modifier flags!
    public func pollModifierToggle(
        controllerKey: ObjectIdentifier?,
        keyCode: UInt16,
        flags: NSEvent.ModifierFlags
    ) -> ChimeResponse? {
        guard let handle else { return nil }
        let now = ProcessInfo.processInfo.systemUptime
        let rawFlags = flags.rawValue
        let cleanMask = flags.intersection(.deviceIndependentFlagsMask).rawValue
        // Deduplicate if both IMKInputController and global monitor deliver the same event
        // (comparing device-independent flags because IMK IPC often strips low device bits).
        if (now - lastProcessedFlagsUptime) < 0.055 && cleanMask == lastProcessedFlagsMask {
            return nil
        }
        lastProcessedFlagsUptime = now
        lastProcessedFlagsMask = cleanMask

        let toggleRule = cachedConfig.english_toggle_key
        if toggleRule == "none" {
            shiftDownUptime = nil
            return nil
        }

        let hasOtherMod = flags.contains(.command) || flags.contains(.control) || flags.contains(.option)
        if hasOtherMod {
            keyOrMouseDuringShift = true
            shiftDownUptime = nil
            return nil
        }

        let sid = (controllerKey ?? activeControllerKey).flatMap { slots[$0]?.sessionId } ?? 0

        if toggleRule == "caps_lock" {
            guard keyCode == 0x39 else { return nil }
            if (now - lastModeToggleUptime) < 0.15 { return nil }
            lastModeToggleUptime = now
            let resp = decodeAndFree(chime_toggle_english(handle, sid))
            inlineEnglishModeGlobal = resp.inline_english_mode
            return resp
        }

        // NX_DEVICELSHIFTKEYMASK = 0x02, NX_DEVICERSHIFTKEYMASK = 0x04
        let isLeftDeviceBit = (rawFlags & 0x0002) != 0
        let isRightDeviceBit = (rawFlags & 0x0004) != 0
        let shiftDown = flags.contains(.shift)

        if shiftDown {
            let detectedSide: UInt16 = if keyCode == 0x38 || isLeftDeviceBit {
                0x38
            } else if keyCode == 0x3C || isRightDeviceBit {
                0x3C
            } else {
                0
            }
            let allowed: Bool = switch toggleRule {
            case "left_shift":
                detectedSide == 0x38
            case "right_shift":
                detectedSide == 0x3C
            default:
                true
            }
            if allowed {
                shiftDownUptime = now
                shiftDownSide = detectedSide
                keyOrMouseDuringShift = false
            } else {
                shiftDownUptime = nil
                keyOrMouseDuringShift = true
            }
            return nil
        }

        // Shift was just released (`!shiftDown`)
        guard let downTs = shiftDownUptime else {
            return nil
        }
        let prevSide = shiftDownSide ?? 0
        shiftDownUptime = nil
        shiftDownSide = nil

        if keyCode != 0 && keyCode != 0xFFFF && keyCode != 0x38 && keyCode != 0x3C {
            return nil
        }
        if toggleRule == "left_shift" && keyCode == 0x3C && prevSide != 0x38 {
            return nil
        }
        if toggleRule == "right_shift" && keyCode == 0x38 && prevSide != 0x3C {
            return nil
        }

        let cleanTap = !keyOrMouseDuringShift && (now - downTs) < 0.35
        guard cleanTap else { return nil }
        // Hard 150ms toggle cooldown prevents double-firing across IMK + GlobalMonitor IPC.
        guard (now - lastModeToggleUptime) >= 0.15 else { return nil }
        lastModeToggleUptime = now

        let resp = decodeAndFree(chime_toggle_english(handle, sid))
        inlineEnglishModeGlobal = resp.inline_english_mode
        return resp
    }

    public func toggleEnglish(controllerKey: ObjectIdentifier?) -> ChimeResponse {
        guard let handle else { return .unhandled }
        let sid = (controllerKey ?? activeControllerKey).flatMap { slots[$0]?.sessionId } ?? 0
        let resp = decodeAndFree(chime_toggle_english(handle, sid))
        inlineEnglishModeGlobal = resp.inline_english_mode
        syncMenuBarEnBadge()
        return resp
    }

    public func togglePunctuation(controllerKey: ObjectIdentifier?) -> ChimeResponse {
        guard let handle else { return .unhandled }
        let sid = (controllerKey ?? activeControllerKey).flatMap { slots[$0]?.sessionId } ?? 0
        let rawPtr = chime_toggle_punctuation(handle, sid)
        return decodeAndFree(rawPtr)
    }

    public func togglePanguSpacing(controllerKey: ObjectIdentifier?) -> ChimeResponse {
        guard let handle else { return .unhandled }
        let sid = (controllerKey ?? activeControllerKey).flatMap { slots[$0]?.sessionId } ?? 0
        let rawPtr = chime_toggle_pangu_spacing(handle, sid)
        return decodeAndFree(rawPtr)
    }

    @discardableResult
    public func toggleEnglishAutoCorrect() -> Bool {
        var cfg = getConfig() ?? .default
        cfg.english_auto_correct.toggle()
        _ = setConfig(cfg)
        return cfg.english_auto_correct
    }

    public func toggleGrid(controllerKey: ObjectIdentifier) -> ChimeResponse {
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else {
            return .unhandled
        }
        let rawPtr = chime_toggle_grid(handle, slot.sessionId)
        return decodeAndFree(rawPtr)
    }

    public func processKey(
        controllerKey: ObjectIdentifier,
        asciiChar: UInt8,
        keyCode: UInt16,
        shift: Bool,
        hasCommandOrCtrl: Bool,
        secureInput: Bool
    ) -> ChimeResponse {
        activeControllerKey = controllerKey
        keyOrMouseDuringShift = true
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else {
            return .unhandled
        }

        let rawPtr = chime_process_key(
            handle,
            slot.sessionId,
            asciiChar,
            keyCode,
            shift,
            hasCommandOrCtrl,
            secureInput
        )
        let resp = decodeAndFree(rawPtr)
        inlineEnglishModeGlobal = resp.inline_english_mode
        syncMenuBarEnBadge()
        if !resp.candidates.isEmpty {
            scheduleAsyncRerankPoll(controllerKey: controllerKey, sessionId: slot.sessionId)
        }
        return resp
    }

    public func selectCandidateFromPanel(controllerKey: ObjectIdentifier, index: Int) -> ChimeResponse {
        guard let slot = slots[controllerKey], let handle, slot.sessionId != 0 else {
            return .unhandled
        }
        let rawPtr = chime_select_candidate(handle, slot.sessionId, index)
        return decodeAndFree(rawPtr)
    }

    public func updateCandidatePanel(
        controllerKey: ObjectIdentifier,
        response: ChimeResponse,
        observedCaretRect: NSRect?
    ) {
        var caret = NSRect(x: 480, y: 360, width: 1, height: 18)
        if var slot = slots[controllerKey] {
            if let observedCaretRect {
                slot.lastCaretRect = observedCaretRect
                slot.hasObservedCaret = true
                slots[controllerKey] = slot
            }
            caret = slot.lastCaretRect
        }
        CandidatePanel.shared.update(response: response, caretRect: caret)
    }

    public func recordObservedCaretRect(controllerKey: ObjectIdentifier, rect: NSRect?) {
        guard let rect, var slot = slots[controllerKey] else { return }
        slot.lastCaretRect = rect
        slot.hasObservedCaret = true
        slots[controllerKey] = slot
    }

    /// Share the marked-text lifecycle between keyboard and candidate-panel selection.
    func nextMarkUpdate(controllerKey: ObjectIdentifier, resp: ChimeResponse) -> ClientMarkUpdate {
        let update = ClientMarkUpdate(previouslyMarked: hadMarkedPreedit(for: controllerKey),
                                      committed: resp.committed_text, preedit: resp.preedit_text)
        setHadMarkedPreedit(for: controllerKey, value: !resp.preedit_text.isEmpty)
        return update
    }

    public func applyResponseToStoredClient(controllerKey: ObjectIdentifier, resp: ChimeResponse) {
        guard let slot = slots[controllerKey], let client = slot.lastClientObject as? IMKTextInput else {
            return
        }
        if !resp.committed_text.isEmpty {
            client.insertText(
                resp.committed_text,
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )
        }
        nextMarkUpdate(controllerKey: controllerKey, resp: resp).apply(to: client)
        if resp.candidates.isEmpty && resp.ghost_text.isEmpty {
            CandidatePanel.shared.hide()
        } else {
            CandidatePanel.shared.update(response: resp, caretRect: slot.lastCaretRect)
        }
    }

    public func getConfig() -> ChimeConfigDTO? {
        guard let handle else { return nil }
        let rawPtr = chime_get_config_json(handle)
        return decodeGenericAndFree(rawPtr, as: ChimeConfigDTO.self)
    }

    @discardableResult
    public func setConfig(_ config: ChimeConfigDTO) -> Bool {
        guard let handle,
              let data = try? encoder.encode(config),
              let jsonStr = String(data: data, encoding: .utf8)
        else {
            return false
        }
        let ok = jsonStr.withCString { cStr in
            chime_set_config_json(handle, cStr)
        }
        if ok {
            cachedConfig = config
        }
        return ok
    }

    public func getStatus() -> ChimeStatusDTO? {
        guard let handle else { return nil }
        let rawPtr = chime_get_status_json(handle)
        return decodeGenericAndFree(rawPtr, as: ChimeStatusDTO.self)
    }

    public func importAsset(url: URL) -> Int64 {
        guard let handle else { return -1 }
        return url.path.withCString { cStr in
            chime_import_asset(handle, cStr)
        }
    }

    @discardableResult
    public func removePack(id: String) -> Bool {
        guard let handle else { return false }
        return id.withCString { cStr in
            chime_remove_pack(handle, cStr)
        }
    }

    public func reloadAssets() {
        guard let handle else { return }
        chime_reload_assets(handle)
        if let cfg = getConfig() {
            cachedConfig = cfg
        }
    }

    public func clearUserHabits() {
        guard let handle else { return }
        chime_clear_user_habits(handle)
    }

    private func scheduleAsyncRerankPoll(controllerKey: ObjectIdentifier, sessionId: UInt64) {
        pollTasks[controllerKey]?.cancel()
        pollTasks[controllerKey] = Task { @MainActor [weak self] in
            for delayMs in [12, 20, 32, 55, 90] {
                try? await Task.sleep(for: .milliseconds(delayMs))
                if Task.isCancelled { return }
                guard let self, let handle = self.handle, let slot = self.slots[controllerKey], slot.sessionId == sessionId else {
                    return
                }
                if let rawPtr = chime_poll_async_rerank(handle, sessionId) {
                    let updated = self.decodeAndFree(rawPtr)
                    if !updated.candidates.isEmpty || !updated.ghost_text.isEmpty {
                        CandidatePanel.shared.update(
                            response: updated,
                            caretRect: slot.lastCaretRect,
                            preservePosition: true
                        )
                    }
                    return
                }
            }
        }
    }

    private func decodeAndFree(_ rawPtr: UnsafeMutablePointer<CChar>?) -> ChimeResponse {
        decodeGenericAndFree(rawPtr, as: ChimeResponse.self) ?? .unhandled
    }

    private func decodeGenericAndFree<T: Decodable>(_ rawPtr: UnsafeMutablePointer<CChar>?, as type: T.Type) -> T? {
        guard let rawPtr else { return nil }
        defer { chime_string_free(rawPtr) }
        let jsonStr = String(cString: rawPtr)
        guard let data = jsonStr.data(using: .utf8) else { return nil }
        return try? decoder.decode(type, from: data)
    }
}
