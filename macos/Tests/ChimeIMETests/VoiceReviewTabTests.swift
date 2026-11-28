import AppKit
import XCTest
@testable import ChimeIME

final class VoiceReviewTabTests: XCTestCase {
    private func key(_ code: CGKeyCode = 0x30, down: Bool = true, repeatDown: Bool = false,
                     flags: CGEventFlags = []) throws -> CGEvent {
        let event = try XCTUnwrap(CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down))
        event.flags = flags
        event.setIntegerValueField(.keyboardEventAutorepeat, value: repeatDown ? 1 : 0)
        return event
    }

    // In-process host harness: an editable composer followed by an attachment control. Route
    // synthetic Quartz events through the production filter policy. Never post keys to the user.
    @MainActor func testConfirmationDoesNotDeliverFocusTraversalToComposer() throws {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 90),
                              styleMask: .titled, backing: .buffered, defer: false)
        let composer = NSTextView(frame: NSRect(x: 0, y: 30, width: 200, height: 60))
        let attachment = NSButton(title: "Attach files", target: nil, action: nil)
        window.contentView?.addSubview(composer); window.contentView?.addSubview(attachment)
        XCTAssertTrue(window.makeFirstResponder(composer))
        var router = VoiceReviewTab(), reviewing = true, adopted = 0, hostEvents = 0
        func dispatch(_ event: CGEvent) {
            switch router.route(event.type, event, reviewing: reviewing) {
            case .adopt: adopted += 1; reviewing = false
            case .consume: break
            case .pass:
                hostEvents += 1
                if event.type == .keyDown { _ = window.makeFirstResponder(attachment) }
            }
        }
        dispatch(try key()); dispatch(try key(repeatDown: true)); dispatch(try key(down: false))
        XCTAssertEqual(adopted, 1); XCTAssertEqual(hostEvents, 0)
        XCTAssertTrue(window.firstResponder === composer)
        dispatch(try key()); dispatch(try key(down: false))
        XCTAssertTrue(window.firstResponder === attachment); XCTAssertEqual(hostEvents, 2)
    }

    func testEnterShortcutsAndInvisibleOrExpiredSuggestionsPass() throws {
        var router = VoiceReviewTab()
        for keyCode: CGKeyCode in [0x24, 0x4C] {
            XCTAssertEqual(router.route(.keyDown, try key(keyCode), reviewing: true), .pass)
        }
        for flags: CGEventFlags in [.maskShift, .maskCommand, .maskControl, .maskAlternate, .maskSecondaryFn] {
            XCTAssertEqual(router.route(.keyDown, try key(flags: flags), reviewing: true), .pass)
        }
        XCTAssertEqual(router.route(.keyDown, try key(), reviewing: false), .pass)
        XCTAssertEqual(router.route(.keyDown, try key(repeatDown: true), reviewing: true), .pass)
        XCTAssertEqual(router.route(.keyUp, try key(down: false), reviewing: false), .pass)
    }

    func testMissingReleaseDoesNotStealNewPress() throws {
        var router = VoiceReviewTab()
        XCTAssertEqual(router.route(.keyDown, try key(), reviewing: true), .adopt)
        XCTAssertEqual(router.route(.keyUp, try key(0x24, down: false), reviewing: false), .pass)
        XCTAssertEqual(router.route(.keyDown, try key(repeatDown: true), reviewing: false), .consume)
        XCTAssertEqual(router.route(.keyDown, try key(), reviewing: false), .pass)
        XCTAssertEqual(router.route(.keyUp, try key(down: false), reviewing: false), .pass)
    }
}
