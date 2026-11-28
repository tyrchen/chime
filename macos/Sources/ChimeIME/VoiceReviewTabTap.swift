import AppKit
import ApplicationServices
import Carbon

/// A short-lived filter for the *target process*, upstream of its responder/DOM key handlers.
/// No event posting, no system-wide tap and no permission prompt. Installed only for a visible
/// review. Once Tab is claimed, keep filtering its repeats/release after the HUD is gone.
@MainActor
final class VoiceReviewTabTap {
    private var port: CFMachPort?
    private var source: CFRunLoopSource?
    private var latch = VoiceReviewTab()
    private var hostPID: pid_t = 0
    private var adopt: (() -> Void)?
    private var unavailable: (() -> Void)?
    private var cleanup: Task<Void, Never>?

    func begin(bundleID: String?, adopt: @escaping () -> Void, unavailable: @escaping () -> Void) -> Bool {
        stop()
        guard AXIsProcessTrusted(), !IsSecureEventInputEnabled(), let bundleID, !bundleID.isEmpty,
              let host = NSWorkspace.shared.frontmostApplication, host.bundleIdentifier == bundleID else { return false }
        let mask: CGEventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
        guard let tap = CGEvent.tapCreateForPid(pid: host.processIdentifier, place: .headInsertEventTap,
            options: .defaultTap, eventsOfInterest: mask, callback: { _, type, event, context in
                guard let context else { return Unmanaged.passUnretained(event) }
                let owner = Unmanaged<VoiceReviewTabTap>.fromOpaque(context).takeUnretainedValue()
                // This port's source is scheduled exclusively on the main run loop.
                return MainActor.assumeIsolated { owner.filter(type, event) } ? nil : Unmanaged.passUnretained(event)
            }, userInfo: Unmanaged.passUnretained(self).toOpaque()),
            let src = CFMachPortCreateRunLoopSource(nil, tap, 0) else { return false }
        port = tap; source = src; hostPID = host.processIdentifier
        self.adopt = adopt; self.unavailable = unavailable
        CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        guard CGEvent.tapIsEnabled(tap: tap) else { stop(); return false }
        return true
    }

    private func filter(_ type: CGEventType, _ event: CGEvent) -> Bool {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            let notify = unavailable
            stop(); notify?()
            return false
        }
        guard let key = UInt16(exactly: event.getIntegerValueField(.keyboardEventKeycode)),
              key == VoiceReviewTab.key else { return false }
        if type == .keyDown && (IsSecureEventInputEnabled() ||
            NSWorkspace.shared.frontmostApplication?.processIdentifier != hostPID) { return false }
        switch latch.route(type, event, reviewing: adopt != nil) {
        case .pass:
            if adopt == nil && latch.held == nil { stop() } // release not delivered; pass the next new press
            return false
        case .consume:
            if latch.held == nil && adopt == nil { stop() }
            return true
        case .adopt:
            let confirm = adopt
            adopt = nil
            // Never run IMK text mutation inside a Quartz event-tap callback.
            DispatchQueue.main.async { confirm?() }
            return true
        }
    }

    func endReview() {
        adopt = nil; unavailable = nil
        guard latch.held != nil else { stop(); return }
        // If focus left the process, its keyUp may never arrive. Bound the drain lifetime.
        cleanup?.cancel()
        cleanup = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(15))
            guard !Task.isCancelled, let self, let key = self.latch.held else { return }
            // A long-held Tab must not turn into auto-repeat focus traversal after a timer expires.
            if NSWorkspace.shared.frontmostApplication?.processIdentifier == self.hostPID,
               CGEventSource.keyState(.combinedSessionState, key: key) {
                self.endReview()
            } else { self.stop() }
        }
    }

    func stop() {
        cleanup?.cancel(); cleanup = nil
        adopt = nil; unavailable = nil; latch = VoiceReviewTab()
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let port { CFMachPortInvalidate(port) }
        source = nil; port = nil
    }
}
