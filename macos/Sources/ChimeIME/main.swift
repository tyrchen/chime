import AppKit
import Carbon
import InputMethodKit

enum ChimeInputSourceRegistrar {
    static let bundleId = "app.tyrchen.inputmethod.Chime"
    static let canonicalModeId = "app.tyrchen.inputmethod.Chime.Pinyin"

    @discardableResult
    static func registerAndEnable(bundleURL: URL, selectMode: Bool) -> Bool {
        let regStatus = TISRegisterInputSource(bundleURL as CFURL)
        if regStatus != noErr {
            fputs("Chime: TISRegisterInputSource returned \(regStatus) for \(bundleURL.path)\n", stderr)
        }

        let parentFilter = [kTISPropertyInputSourceID as String: bundleId] as CFDictionary
        let modeFilter = [kTISPropertyInputSourceID as String: canonicalModeId] as CFDictionary

        // Poll up to 3.0s for HIServices / TextInputMenuAgent to index both parent bundle and mode.
        var foundMode: TISInputSource?
        for _ in 0..<15 {
            if let parents = TISCreateInputSourceList(parentFilter, true)?.takeRetainedValue() as? [TISInputSource],
               let parent = parents.first {
                _ = TISEnableInputSource(parent)
            }
            if let modes = TISCreateInputSourceList(modeFilter, true)?.takeRetainedValue() as? [TISInputSource],
               let mode = modes.first {
                foundMode = mode
                break
            }
            Thread.sleep(forTimeInterval: 0.2)
        }

        guard let mode = foundMode else {
            fputs("Chime: input mode \(canonicalModeId) not yet indexed after registration\n", stderr)
            return false
        }

        let enStatus = TISEnableInputSource(mode)
        if selectMode {
            _ = TISSelectInputSource(mode)
        }

        let center = DistributedNotificationCenter.default()
        center.postNotificationName(
            NSNotification.Name(kTISNotifyEnabledKeyboardInputSourcesChanged as String),
            object: nil,
            deliverImmediately: true
        )
        return enStatus == noErr
    }
}

@MainActor
enum ChimeBootstrap {
    static var server: IMKServer?

    static func run() {
        let args = CommandLine.arguments
        if let regIdx = args.firstIndex(of: "--register") {
            let targetPath = (regIdx + 1 < args.count) ? args[regIdx + 1] : Bundle.main.bundleURL.path
            let url = URL(fileURLWithPath: targetPath, isDirectory: true)
            _ = ChimeInputSourceRegistrar.registerAndEnable(bundleURL: url, selectMode: true)
            exit(0)
        }

        // Idempotent self-registration if a user dragged Chime.app from .zip and opened it directly.
        _ = TISRegisterInputSource(Bundle.main.bundleURL as CFURL)

        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let bundleId = Bundle.main.bundleIdentifier ?? ChimeInputSourceRegistrar.bundleId
        server = IMKServer(name: "Chime_1_Connection", bundleIdentifier: bundleId)
        app.run()
    }
}

ChimeBootstrap.run()
