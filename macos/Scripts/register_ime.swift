import Carbon
import Foundation

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: swift register_ime.swift <path/to/Chime.app>\n", stderr)
    exit(1)
}

let bundleURL = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let status = TISRegisterInputSource(bundleURL as CFURL)
if status != noErr {
    fputs("Warning: TISRegisterInputSource returned OSStatus \(status)\n", stderr)
} else {
    print("Registered input source bundle with TIS: \(bundleURL.path)")
}
