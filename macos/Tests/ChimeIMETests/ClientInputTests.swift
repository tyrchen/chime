import AppKit
import XCTest
@testable import ChimeIME

final class ClientInputTests: XCTestCase {
    func testClickToFocusIsNotAKey() throws {
        for type: NSEvent.EventType in [.leftMouseDown, .rightMouseDown] {
            let click = try XCTUnwrap(NSEvent.mouseEvent(with: type, location: .zero, modifierFlags: [],
                timestamp: 0, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
            XCTAssertNil(ChimeKeyInput(click))
        }
    }

    func testModifiersDoNotReadCharactersButKeyDownStillDoes() throws {
        let ctrl = try XCTUnwrap(NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: .control,
            timestamp: 0, windowNumber: 0, context: nil, characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 59))
        let modifier = try XCTUnwrap(ChimeKeyInput(ctrl))
        XCTAssertEqual(modifier.keyCode, 59)
        XCTAssertEqual(modifier.ascii, 0)
        XCTAssertFalse(modifier.isControlPeriod)
        let n = try XCTUnwrap(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
            timestamp: 0, windowNumber: 0, context: nil, characters: "n", charactersIgnoringModifiers: "n", isARepeat: false, keyCode: 45))
        XCTAssertEqual(ChimeKeyInput(n)?.ascii, 110)
    }

    func testCommittingIsNotCancellingTheComposition() {
        XCTAssertEqual(ClientMarkUpdate(previouslyMarked: false, committed: "", preedit: "ni"), .set("ni"))
        XCTAssertEqual(ClientMarkUpdate(previouslyMarked: true, committed: "你", preedit: ""), .none)
        XCTAssertEqual(ClientMarkUpdate(previouslyMarked: true, committed: "你", preedit: "hao"), .set("hao"))
        XCTAssertEqual(ClientMarkUpdate(previouslyMarked: true, committed: "", preedit: ""), .clear)
        XCTAssertEqual(ClientMarkUpdate(previouslyMarked: false, committed: "", preedit: ""), .none)
    }
}
