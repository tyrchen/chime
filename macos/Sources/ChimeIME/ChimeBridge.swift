import AppKit
import CChimeFFI
import Foundation

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
    public let generation_id: UInt64

    public static let unhandled = ChimeResponse(
        handled: false,
        committed_text: "",
        preedit_text: "",
        candidates: [],
        generation_id: 0
    )
}

/// MainActor-isolated coordinator owning the Rust `ChimeEngineHandle` and lightweight per-controller
/// metadata keyed by `ObjectIdentifier` (leaving `IMKInputController` with zero stored properties).
@MainActor
public final class ChimeEngineBridge {
    public static let shared = ChimeEngineBridge()

    private struct ControllerSlot {
        var sessionId: UInt64
        var lastCaretRect: NSRect
        var leftShiftDownTs: TimeInterval?
        var keyPressedDuringShift: Bool
    }

    private var handle: OpaquePointer?
    private let decoder = JSONDecoder()
    private var slots: [ObjectIdentifier: ControllerSlot] = [:]

    private init() {
        self.handle = chime_engine_new()
    }

    public func activate(controllerKey: ObjectIdentifier, appBundleId: String) {
        if slots[controllerKey] != nil {
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
            keyPressedDuringShift: false
        )
    }

    public func deactivate(controllerKey: ObjectIdentifier) {
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

    public func pollLeftShiftTap(
        controllerKey: ObjectIdentifier,
        keyCode: UInt16,
        shiftDown: Bool,
        timestamp: TimeInterval
    ) -> ChimeResponse? {
        // Left Shift virtual keyCode on macOS is 0x38 (56).
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

    public func processKey(
        controllerKey: ObjectIdentifier,
        asciiChar: UInt8,
        keyCode: UInt16,
        shift: Bool,
        hasCommandOrCtrl: Bool,
        secureInput: Bool
    ) -> ChimeResponse {
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
        return decodeAndFree(rawPtr)
    }

    public func updateCandidatePanel(
        controllerKey: ObjectIdentifier,
        candidates: [ChimeCandidate],
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
        CandidatePanel.shared.update(candidates: candidates, caretRect: caret)
    }

    private func decodeAndFree(_ rawPtr: UnsafeMutablePointer<CChar>?) -> ChimeResponse {
        guard let rawPtr else { return .unhandled }
        defer { chime_string_free(rawPtr) }
        let jsonStr = String(cString: rawPtr)
        guard let data = jsonStr.data(using: .utf8),
              let decoded = try? decoder.decode(ChimeResponse.self, from: data)
        else {
            return .unhandled
        }
        return decoded
    }
}
