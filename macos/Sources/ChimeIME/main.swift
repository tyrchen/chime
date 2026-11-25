import AppKit
import Carbon
import InputMethodKit

enum ChimeInputSourceRegistrar {
    static let bundleId = "app.tyrchen.inputmethod.Chime"
    static let canonicalModeId = "app.tyrchen.inputmethod.Chime.Pinyin"
    static let initialAutoEnableDefaultsKey = "chime.didInitialAutoEnable.v1"

    /// Registers `bundleURL` with TextInputSources and optionally enables/selects `Chime 拼音`.
    /// Must be called from the long-lived `Chime` IMKServer process so macOS keeps the dynamic
    /// TIS registration bound across the session without requiring logout.
    @discardableResult
    static func registerStep(bundleURL: URL, enableAndSelect: Bool) -> Bool {
        let regStatus = TISRegisterInputSource(bundleURL as CFURL)
        if regStatus != noErr {
            NSLog("Chime: TISRegisterInputSource returned %d for %@", regStatus, bundleURL.path)
        }

        let parentFilter = [kTISPropertyInputSourceID as String: bundleId] as CFDictionary
        if let parents = TISCreateInputSourceList(parentFilter, true)?.takeRetainedValue() as? [TISInputSource],
           let parent = parents.first
        {
            if enableAndSelect {
                _ = TISEnableInputSource(parent)
            }
        }

        let modeFilter = [kTISPropertyInputSourceID as String: canonicalModeId] as CFDictionary
        guard let modes = TISCreateInputSourceList(modeFilter, true)?.takeRetainedValue() as? [TISInputSource],
              let mode = modes.first
        else {
            return false
        }

        if enableAndSelect {
            let enStatus = TISEnableInputSource(mode)
            let selStatus = TISSelectInputSource(mode)
            NSLog("Chime: enabled=%d selected=%d for %@", enStatus, selStatus, canonicalModeId)
            if enStatus == noErr {
                UserDefaults.standard.set(true, forKey: initialAutoEnableDefaultsKey)
            }
            DistributedNotificationCenter.default().postNotificationName(
                NSNotification.Name(kTISNotifyEnabledKeyboardInputSourcesChanged as String),
                object: nil,
                deliverImmediately: true
            )
            return enStatus == noErr
        }

        return true
    }

    @discardableResult
    static func registerBlockingCLI(bundleURL: URL, selectMode: Bool) -> Bool {
        for _ in 0..<15 {
            if registerStep(bundleURL: bundleURL, enableAndSelect: selectMode) {
                return true
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return false
    }

    @MainActor
    static func scheduleDaemonRegistration(forceEnable: Bool) {
        let bundleURL = Bundle.main.bundleURL
        let shouldEnable = forceEnable || !UserDefaults.standard.bool(forKey: initialAutoEnableDefaultsKey)

        // Immediate pass before RunLoop spins, followed by staggered passes on the active Aqua RunLoop
        // so HIServices / TextInputMenuAgent / System Settings always index and retain the bundle.
        if registerStep(bundleURL: bundleURL, enableAndSelect: shouldEnable), !shouldEnable {
            return
        }

        let delays: [TimeInterval] = [0.25, 0.9, 2.0, 4.0]
        for delay in delays {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                let stillNeedsEnable = forceEnable || !UserDefaults.standard.bool(forKey: initialAutoEnableDefaultsKey)
                _ = registerStep(bundleURL: bundleURL, enableAndSelect: stillNeedsEnable)
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

        let forceEnable = args.contains("--activate-on-install")

        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let bundleId = Bundle.main.bundleIdentifier ?? ChimeInputSourceRegistrar.bundleId
        server = IMKServer(name: "Chime_1_Connection", bundleIdentifier: bundleId)
        ChimeInputSourceRegistrar.scheduleDaemonRegistration(forceEnable: forceEnable)
        app.run()
    }
}

ChimeBootstrap.run()
