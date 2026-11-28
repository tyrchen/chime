import Foundation
import XCTest

final class InputMethodRegistrationTests: XCTestCase {
    func testAdvertisedEndpointUsesRegisteredBundleIdentifier() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: packageRoot.appendingPathComponent("Resources/Info.plist"))
        let plist = try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
        let identifier = try XCTUnwrap(plist["CFBundleIdentifier"] as? String)
        XCTAssertEqual(plist["InputMethodConnectionName"] as? String, identifier + "_Connection")
    }
}
