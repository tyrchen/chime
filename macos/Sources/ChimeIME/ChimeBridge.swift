import AppKit
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

    public static let `default` = ChimeConfigDTO(
        scheme: "full",
        fuzzy: .default,
        pangu_spacing: true,
        chinese_punctuation: true,
        english_toggle_key: "both_shift",
        shift_commit_mode: "commit_and_toggle",
        uppercase_triggers_english: true,
        show_status_toast: true,
        minus_equal_paging: true,
        bracket_paging: false,
        comma_period_paging: false,
        semicolon_apostrophe_select: false,
        record_training_samples: false,
        anti_jitter_ms: 60,
        page_size: 5,
        candidate_font_size: 15,
        candidate_layout: "horizontal"
    )
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
}

/// MainActor-isolated coordinator owning the persistent Rust `ChimeEngineHandle` and lightweight
/// per-controller metadata keyed by `ObjectIdentifier`.
@MainActor
public final class ChimeEngineBridge {
    public static let shared = ChimeEngineBridge()

    private struct ControllerSlot {
        var sessionId: UInt64
        var lastCaretRect: NSRect
        var shiftDownTs: TimeInterval?
        var shiftDownKeyCode: UInt16?
        var keyOrOtherModDuringShift: Bool
        var inlineEnglishMode: Bool
        weak var lastClientObject: AnyObject?
    }

    private var handle: OpaquePointer?
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private var slots: [ObjectIdentifier: ControllerSlot] = [:]
    public var activeControllerKey: ObjectIdentifier?

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
                let caret = self.slots[key]?.lastCaretRect ?? NSRect(x: 480, y: 360, width: 1, height: 18)
                StatusToastPanel.shared.show(
                    badge: resp.chinese_punctuation ? "中。" : "英.",
                    caretRect: caret
                )
            }
        }

        CandidatePanel.shared.onToggleEnglish = { [weak self] in
            guard let self, let key = self.activeControllerKey else { return }
            let resp = self.toggleEnglish(controllerKey: key)
            self.applyResponseToStoredClient(controllerKey: key, resp: resp)
            if resp.show_status_toast {
                let caret = self.slots[key]?.lastCaretRect ?? NSRect(x: 480, y: 360, width: 1, height: 18)
                StatusToastPanel.shared.show(
                    badge: resp.inline_english_mode ? "英" : "中",
                    caretRect: caret
                )
            }
        }
    }

    public func isInlineEnglish(controllerKey: ObjectIdentifier?) -> Bool {
        guard let key = controllerKey ?? activeControllerKey else { return false }
        return slots[key]?.inlineEnglishMode ?? false
    }

    public func caretRect(for controllerKey: ObjectIdentifier) -> NSRect {
        slots[controllerKey]?.lastCaretRect ?? NSRect(x: 480, y: 360, width: 1, height: 18)
    }

    public func activate(controllerKey: ObjectIdentifier, appBundleId: String, client: AnyObject?) {
        activeControllerKey = controllerKey
        if var existing = slots[controllerKey] {
            existing.lastClientObject = client
            slots[controllerKey] = existing
            return
        }
        guard let handle else { return }
        let sid = appBundleId.withCString { cStr in
            chime_session_create(handle, cStr)
        }
        slots[controllerKey] = ControllerSlot(
            sessionId: sid,
            lastCaretRect: NSRect(x: 480, y: 360, width: 1, height: 18),
            shiftDownTs: nil,
            shiftDownKeyCode: nil,
            keyOrOtherModDuringShift: false,
            inlineEnglishMode: false,
            lastClientObject: client
        )
    }

    public func deactivate(controllerKey: ObjectIdentifier) {
        if activeControllerKey == controllerKey {
            activeControllerKey = nil
        }
        CandidatePanel.shared.hide()
        StatusToastPanel.shared.hide()
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

    /// Detects clean modifier taps (`both_shift`, `left_shift`, `right_shift`, `caps_lock`) without
    /// misfiring on multi-modifier system shortcuts (`Cmd+Shift+4`, `Ctrl+Shift`, `Opt+Shift`).
    public func pollModifierToggle(
        controllerKey: ObjectIdentifier,
        keyCode: UInt16,
        flags: NSEvent.ModifierFlags,
        timestamp: TimeInterval
    ) -> ChimeResponse? {
        guard var slot = slots[controllerKey], let handle else { return nil }
        let toggleRule = getConfig()?.english_toggle_key ?? "both_shift"
        if toggleRule == "none" {
            slot.shiftDownTs = nil
            slots[controllerKey] = slot
            return nil
        }

        let hasOtherMod = flags.contains(.command) || flags.contains(.control) || flags.contains(.option)
        if hasOtherMod {
            slot.keyOrOtherModDuringShift = true
            slot.shiftDownTs = nil
            slots[controllerKey] = slot
            return nil
        }

        if toggleRule == "caps_lock" {
            guard keyCode == 0x39 else { return nil }
            let resp = decodeAndFree(chime_toggle_english(handle, slot.sessionId))
            slot.inlineEnglishMode = resp.inline_english_mode
            slots[controllerKey] = slot
            return resp
        }

        let allowedShift: Bool = {
            switch toggleRule {
            case "left_shift":
                return keyCode == 0x38
            case "right_shift":
                return keyCode == 0x3C
            default:
                return keyCode == 0x38 || keyCode == 0x3C
            }
        }()
        guard allowedShift else {
            slot.keyOrOtherModDuringShift = true
            slots[controllerKey] = slot
            return nil
        }

        let shiftDown = flags.contains(.shift)
        if shiftDown {
            slot.shiftDownTs = timestamp
            slot.shiftDownKeyCode = keyCode
            slot.keyOrOtherModDuringShift = false
            slots[controllerKey] = slot
            return nil
        }

        guard let downTs = slot.shiftDownTs, slot.shiftDownKeyCode == keyCode else {
            slot.shiftDownTs = nil
            slots[controllerKey] = slot
            return nil
        }
        slot.shiftDownTs = nil
        slot.shiftDownKeyCode = nil
        let cleanTap = !slot.keyOrOtherModDuringShift && (timestamp - downTs) < 0.30
        slots[controllerKey] = slot
        guard cleanTap else { return nil }

        let resp = decodeAndFree(chime_toggle_english(handle, slot.sessionId))
        if var updated = slots[controllerKey] {
            updated.inlineEnglishMode = resp.inline_english_mode
            slots[controllerKey] = updated
        }
        return resp
    }

    public func toggleEnglish(controllerKey: ObjectIdentifier?) -> ChimeResponse {
        guard let key = controllerKey ?? activeControllerKey,
              var slot = slots[key],
              let handle,
              slot.sessionId != 0
        else {
            return .unhandled
        }
        let resp = decodeAndFree(chime_toggle_english(handle, slot.sessionId))
        slot.inlineEnglishMode = resp.inline_english_mode
        slots[key] = slot
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
        guard var slot = slots[controllerKey], let handle, slot.sessionId != 0 else {
            return .unhandled
        }
        slot.keyOrOtherModDuringShift = true
        slots[controllerKey] = slot

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
        if var updated = slots[controllerKey] {
            updated.inlineEnglishMode = resp.inline_english_mode
            slots[controllerKey] = updated
        }
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
                slots[controllerKey] = slot
            }
            caret = slot.lastCaretRect
        }
        CandidatePanel.shared.update(response: response, caretRect: caret)
    }

    public func recordObservedCaretRect(controllerKey: ObjectIdentifier, rect: NSRect?) {
        guard let rect, var slot = slots[controllerKey] else { return }
        slot.lastCaretRect = rect
        slots[controllerKey] = slot
    }

    public func applyResponseToStoredClient(controllerKey: ObjectIdentifier, resp: ChimeResponse) {
        guard let slot = slots[controllerKey], let client = slot.lastClientObject as? IMKTextInput else {
            return
        }
        if !resp.committed_text.isEmpty {
            client.insertText(
                resp.committed_text,
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
        }
        if resp.preedit_text.isEmpty {
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
        } else {
            let nsPreedit = resp.preedit_text as NSString
            client.setMarkedText(
                resp.preedit_text,
                selectionRange: NSRange(location: nsPreedit.length, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
        }
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
        return jsonStr.withCString { cStr in
            chime_set_config_json(handle, cStr)
        }
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

    public func reloadAssets() {
        guard let handle else { return }
        chime_reload_assets(handle)
    }

    public func clearUserHabits() {
        guard let handle else { return }
        chime_clear_user_habits(handle)
    }

    private func scheduleAsyncRerankPoll(controllerKey: ObjectIdentifier, sessionId: UInt64) {
        Task { @MainActor [weak self] in
            for delayMs in [14, 24, 40] {
                try? await Task.sleep(for: .milliseconds(delayMs))
                guard let self, let handle = self.handle, let slot = self.slots[controllerKey], slot.sessionId == sessionId else {
                    return
                }
                if let rawPtr = chime_poll_async_rerank(handle, sessionId) {
                    let updated = self.decodeAndFree(rawPtr)
                    if !updated.candidates.isEmpty || !updated.ghost_text.isEmpty {
                        CandidatePanel.shared.update(response: updated, caretRect: slot.lastCaretRect)
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
