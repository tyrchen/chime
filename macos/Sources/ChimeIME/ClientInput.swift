import AppKit
import InputMethodKit

/// Snapshot before crossing the IMK/main-actor boundary. AppKit raises an ObjC
/// exception (not a Swift error) when a property is read on the wrong event type.
struct ChimeKeyInput {
    let type: NSEvent.EventType
    let keyCode: UInt16
    let flags: NSEvent.ModifierFlags
    let ascii: UInt8
    let isControlPeriod: Bool

    init?(_ event: NSEvent) {
        let type = event.type
        guard type == .keyDown || type == .flagsChanged else { return nil }
        self.type = type
        keyCode = event.keyCode
        flags = event.modifierFlags
        if type == .keyDown {
            let scalar = event.characters?.unicodeScalars.first
            ascii = scalar.flatMap { $0.isASCII ? UInt8($0.value) : nil } ?? 0
            isControlPeriod = flags.contains(.control) && flags.isDisjoint(with: [.command, .option])
                && (keyCode == 0x2F || event.charactersIgnoringModifiers == ".")
        } else {
            ascii = 0
            isControlPeriod = false
        }
    }
}

/// `insertText` ends the old composition. Sending an additional empty mark is
/// a cancellation in Chromium, not an acknowledgement of the commit.
enum ClientMarkUpdate: Equatable {
    case none, clear, set(String)

    func apply(to client: IMKTextInput) {
        let text: String
        switch self {
        case .none: return
        case .clear: text = ""
        case .set(let preedit): text = preedit
        }
        client.setMarkedText(text, selectionRange: NSRange(location: text.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
    }

    init(previouslyMarked: Bool, committed: String, preedit: String) {
        if !preedit.isEmpty { self = .set(preedit) }
        else if previouslyMarked && committed.isEmpty { self = .clear }
        else { self = .none }
    }
}
