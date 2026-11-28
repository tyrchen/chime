import Foundation
import XCTest
@testable import ChimeIME

final class VoiceReviewSiteTests: XCTestCase {
    // Synthetic preedit. NSString/IMK ranges use UTF-16, not Swift Character count.
    private let text = "测🧪试"
    private let mark = NSRange(location: 7, length: 4)
    private let end = NSRange(location: 11, length: 0)
    private let absent = NSRange(location: NSNotFound, length: 0)

    func testHostWithMarkButNoSubstringCanReview() {
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 7, marked: mark, selected: end) { _ in nil }, .marked(mark))
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 7, marked: mark, selected: end) { _ in "其他文本" }, .copyOnly)
    }

    func testHostCommittedOriginalBeforeHUD() {
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 7, marked: absent, selected: end) { range in
            XCTAssertEqual(range, mark); return self.text
        }, .committed(mark))
    }

    func testOpaqueMovedAndEditedFieldsOfferCopyNotReplacement() {
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: nil, marked: absent, selected: absent) { _ in nil }, .copyOnly)
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 7, marked: absent, selected: end) { _ in "其他文本" }, .copyOnly)
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 6, marked: absent, selected: end) { _ in self.text }, .copyOnly)
        XCTAssertEqual(VoiceReviewSite.probe(text: text, start: 7, marked: absent, selected: mark) { _ in self.text }, .copyOnly)
    }
}
