import Carbon
import Foundation

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: swift register_ime.swift <path/to/Chime.app>\n", stderr)
    exit(1)
}

let bundleId = "app.tyrchen.inputmethod.Chime"
let canonicalModeId = "app.tyrchen.inputmethod.Chime.Pinyin"

let bundleURL = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let regStatus = TISRegisterInputSource(bundleURL as CFURL)
if regStatus != noErr {
    fputs("Warning: TISRegisterInputSource returned OSStatus \(regStatus)\n", stderr)
}

let parentFilter = [kTISPropertyInputSourceID as String: bundleId] as CFDictionary
if let parents = TISCreateInputSourceList(parentFilter, true)?.takeRetainedValue() as? [TISInputSource],
   let parent = parents.first {
    _ = TISEnableInputSource(parent)
}

let modeFilter = [kTISPropertyInputSourceID as String: canonicalModeId] as CFDictionary
if let modes = TISCreateInputSourceList(modeFilter, true)?.takeRetainedValue() as? [TISInputSource],
   let mode = modes.first {
    let enStatus = TISEnableInputSource(mode)
    let selStatus = TISSelectInputSource(mode)
    print("Registered & enabled \(canonicalModeId) (enable=\(enStatus), select=\(selStatus))")
} else {
    print("Registered input source bundle with TIS: \(bundleURL.path)")
}
