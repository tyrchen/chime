import AppKit

/// Per-target event tap state. Only a fresh, unmodified Tab may confirm.
/// Once claimed, keep its repeats and release out of the app after the review disappears.
struct VoiceReviewTab {
    enum Action { case pass, adopt, consume }
    private(set) var held: UInt16?

    static let key: UInt16 = 0x30

    /// The process tap maps both .adopt and .consume to a nil Quartz event. Nothing is reposted.
    mutating func route(_ type: CGEventType, _ event: CGEvent, reviewing: Bool) -> Action {
        guard let code = UInt16(exactly: event.getIntegerValueField(.keyboardEventKeycode)), code == Self.key else { return .pass }
        if type == .keyUp { return up(keyCode: code) ? .consume : .pass }
        guard type == .keyDown else { return .pass }
        return down(keyCode: code, flags: NSEvent.ModifierFlags(rawValue: UInt(event.flags.rawValue)),
                    isRepeat: event.getIntegerValueField(.keyboardEventAutorepeat) != 0, reviewing: reviewing)
    }

    private mutating func down(keyCode: UInt16, flags: NSEvent.ModifierFlags, isRepeat: Bool, reviewing: Bool) -> Action {
        guard keyCode == Self.key else { return .pass }
        if isRepeat { return held == keyCode ? .consume : .pass }
        // A lost release must not swallow a second deliberate press of the same key.
        if held == keyCode { held = nil }
        guard reviewing, held == nil, flags.isDisjoint(with: [.shift, .control, .option, .command, .function]) else { return .pass }
        held = keyCode
        return .adopt
    }

    private mutating func up(keyCode: UInt16) -> Bool {
        guard held == keyCode else { return false }
        held = nil
        return true
    }
}
