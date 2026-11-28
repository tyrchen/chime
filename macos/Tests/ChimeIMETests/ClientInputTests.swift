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

    func testTier2ConfigAndStatusDecodingBackwardsCompatible() throws {
        let legacyConfig = try JSONDecoder().decode(ChimeConfigDTO.self, from: Data("{}".utf8))
        XCTAssertFalse(legacyConfig.tier2_enabled)
        XCTAssertEqual(legacyConfig.tier2_model_dir, "")
        XCTAssertEqual(legacyConfig.tier2_margin_threshold, 3.0, accuracy: 1e-6)
        XCTAssertEqual(legacyConfig.tier2_alpha, 0.65, accuracy: 1e-6)

        let tier2JSON = """
        {"tier2_enabled":true,"tier2_model_dir":"/tmp/qwen3-4b","tier2_margin_threshold":4.5,"tier2_alpha":0.75}
        """
        let decoded = try JSONDecoder().decode(ChimeConfigDTO.self, from: Data(tier2JSON.utf8))
        XCTAssertTrue(decoded.tier2_enabled)
        XCTAssertEqual(decoded.tier2_model_dir, "/tmp/qwen3-4b")
        XCTAssertEqual(decoded.tier2_margin_threshold, 4.5, accuracy: 1e-6)
        XCTAssertEqual(decoded.tier2_alpha, 0.75, accuracy: 1e-6)

        let statusJSON = """
        {"scheme":"full","pinyin_keys":10,"word_entries":20,"learned_habits":3,"neural_loaded":true,"tier2_loaded":true,"tier2_model_dir":"/tmp/qwen3-4b"}
        """
        let status = try JSONDecoder().decode(ChimeStatusDTO.self, from: Data(statusJSON.utf8))
        XCTAssertEqual(status.tier2_loaded, true)
        XCTAssertEqual(status.tier2_model_dir, "/tmp/qwen3-4b")

        let legacyStatusJSON = """
        {"scheme":"full","pinyin_keys":10,"word_entries":20,"learned_habits":3,"neural_loaded":true}
        """
        let legacyStatus = try JSONDecoder().decode(ChimeStatusDTO.self, from: Data(legacyStatusJSON.utf8))
        XCTAssertNil(legacyStatus.tier2_loaded)
    }
}
