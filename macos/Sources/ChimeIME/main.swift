import AppKit
import InputMethodKit

@MainActor
enum ChimeBootstrap {
    static var server: IMKServer?

    static func run() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let bundleId = Bundle.main.bundleIdentifier ?? "app.tyrchen.inputmethod.Chime"
        server = IMKServer(name: "Chime_1_Connection", bundleIdentifier: bundleId)
        app.run()
    }
}

ChimeBootstrap.run()
