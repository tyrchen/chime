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
    public let generation_id: UInt64

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
}

/// Full engine configuration mirrored for SwiftUI Settings binding.
public struct ChimeConfigDTO: Codable, Sendable, Equatable {
    public var scheme: String
    public var fuzzy: ChimeFuzzyDTO
    public var pangu_spacing: Bool
    public var chinese_punctuation: Bool
    public var record_training_samples: Bool
    public var anti_jitter_ms: UInt64
    public var page_size: Int
    public var candidate_layout: String
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
        var leftShiftDownTs: TimeInterval?
        var keyPressedDuringShift: Bool
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
        }
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
            leftShiftDownTs: nil,
            keyPressedDuringShift: false,
            lastClientObject: client
        )
    }

    public func deactivate(controllerKey: ObjectIdentifier) {
        if activeControllerKey == controllerKey {
            activeControllerKey = nil
        }
        CandidatePanel.shared.hide()
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

    public func pollLeftShiftTap(
        controllerKey: ObjectIdentifier,
        keyCode: UInt16,
        shiftDown: Bool,
        timestamp: TimeInterval
    ) -> ChimeResponse? {
        guard keyCode == 0x38, var slot = slots[controllerKey], let handle else { return nil }

        if shiftDown {
            slot.leftShiftDownTs = timestamp
            slot.keyPressedDuringShift = false
            slots[controllerKey] = slot
            return nil
        }

        guard let downTs = slot.leftShiftDownTs else { return nil }
        slot.leftShiftDownTs = nil
        let cleanTap = !slot.keyPressedDuringShift && (timestamp - downTs) < 0.25
        slots[controllerKey] = slot
        guard cleanTap else { return nil }

        let rawPtr = chime_toggle_english(handle, slot.sessionId)
        return decodeAndFree(rawPtr)
    }

    public func togglePunctuation(controllerKey: ObjectIdentifier?) -> ChimeResponse {
        guard let handle else { return .unhandled }
        let sid = controllerKey.flatMap { slots[$0]?.sessionId } ?? 0
        let rawPtr = chime_toggle_punctuation(handle, sid)
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
        slot.keyPressedDuringShift = true
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
