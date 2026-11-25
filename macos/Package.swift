// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ChimeIME",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "ChimeIME", targets: ["ChimeIME"])
    ],
    targets: [
        .target(
            name: "CChimeFFI",
            path: "Sources/CChimeFFI",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "ChimeIME",
            dependencies: ["CChimeFFI"],
            path: "Sources/ChimeIME",
            linkerSettings: [
                .linkedLibrary("chime_ffi"),
                .linkedLibrary("sqlite3"),
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("InputMethodKit"),
                .linkedFramework("Carbon"),
                .linkedFramework("CoreML"),
                .linkedFramework("UniformTypeIdentifiers")
            ]
        )
    ]
)
