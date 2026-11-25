import AppKit
import Carbon
import InputMethodKit

enum ChimeInputSourceRegistrar {
    static let bundleId = "app.tyrchen.inputmethod.Chime"
    static let canonicalModeId = "app.tyrchen.inputmethod.Chime.Pinyin"

    private static func isSourceEnabled(_ source: TISInputSource) -> Bool {
        guard let ptr = TISGetInputSourceProperty(source, kTISPropertyInputSourceIsEnabled) else {
            return false
        }
        let flag = Unmanaged<CFBoolean>.fromOpaque(ptr).takeUnretainedValue()
        return CFBooleanGetValue(flag)
    }

    /// Registers `bundleURL` with TextInputSources and synchronizes parent/mode state cleanly.
    ///
    /// On macOS 15+, system anti-keylogger policy keeps a newly installed third-party IME's parent
    /// `TISTypeKeyboardInputMethodModeEnabled` disabled (`IsEnabled = false`) until the user clicks
    /// `Add` once in `System Settings -> Keyboard -> Input Sources -> +`. Crucially, if we enabled
    /// the child mode while the parent stayed disabled, macOS would leave the IME in a half-enabled
    /// zombie state (`parent=false, mode=true`). We explicitly heal/avoid that state so `Chime 拼音`
    /// always appears cleanly and immediately in `+ -> Chinese, Simplified` without logout.
    @discardableResult
    static func registerStep(bundleURL: URL, tryAutoEnable: Bool) -> Bool {
        let regStatus = TISRegisterInputSource(bundleURL as CFURL)
        if regStatus != noErr {
            NSLog("Chime: TISRegisterInputSource returned %d for %@", regStatus, bundleURL.path)
        }

        let parentFilter = [kTISPropertyInputSourceID as String: bundleId] as CFDictionary
        let modeFilter = [kTISPropertyInputSourceID as String: canonicalModeId] as CFDictionary

        guard let parents = TISCreateInputSourceList(parentFilter, true)?.takeRetainedValue() as? [TISInputSource],
              let parent = parents.first,
              let modes = TISCreateInputSourceList(modeFilter, true)?.takeRetainedValue() as? [TISInputSource],
              let mode = modes.first
        else {
            return false
        }

        var parentEnabled = isSourceEnabled(parent)
        if !parentEnabled && tryAutoEnable {
            _ = TISEnableInputSource(parent)
            parentEnabled = isSourceEnabled(parent)
        }

        if parentEnabled {
            if !isSourceEnabled(mode) || tryAutoEnable {
                _ = TISEnableInputSource(mode)
            }
            if tryAutoEnable {
                _ = TISSelectInputSource(mode)
            }
        } else if isSourceEnabled(mode) {
            // Heal half-enabled zombie state from any prior failed programmatic enablement so
            // System Settings -> Keyboard -> Input Sources -> + -> Simplified Chinese shows
            // Chime 拼音 cleanly (`enabled=0, grayOut=0, allow=1`) without requiring logout.
            _ = TISDisableInputSource(mode)
            NSLog("Chime: reset child mode while parent awaits user approval in System Settings")
        }

        DistributedNotificationCenter.default().postNotificationName(
            NSNotification.Name(kTISNotifyEnabledKeyboardInputSourcesChanged as String),
            object: nil,
            deliverImmediately: true
        )
        return true
    }

    @discardableResult
    static func registerBlockingCLI(bundleURL: URL, selectMode: Bool) -> Bool {
        for _ in 0..<15 {
            if registerStep(bundleURL: bundleURL, tryAutoEnable: selectMode) {
                return true
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return false
    }

    @MainActor
    static func scheduleDaemonRegistration(tryAutoEnable: Bool) {
        let bundleURL = Bundle.main.bundleURL
        _ = registerStep(bundleURL: bundleURL, tryAutoEnable: tryAutoEnable)

        let delays: [TimeInterval] = [0.3, 1.0, 2.2]
        for delay in delays {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                _ = registerStep(bundleURL: bundleURL, tryAutoEnable: tryAutoEnable)
            }
        }
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
            _ = ChimeInputSourceRegistrar.registerBlockingCLI(bundleURL: url, selectMode: true)
            exit(0)
        }

        let tryAutoEnable = args.contains("--activate-on-install")

        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let bundleId = Bundle.main.bundleIdentifier ?? ChimeInputSourceRegistrar.bundleId
        server = IMKServer(name: "Chime_1_Connection", bundleIdentifier: bundleId)
        ChimeInputSourceRegistrar.scheduleDaemonRegistration(tryAutoEnable: tryAutoEnable)
        app.run()
    }
}

ChimeBootstrap.run()
