// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ChimeIME",
    platforms: [
        .macOS(.v14)
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
                .linkedLibrary("c++"),
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("InputMethodKit"),
                .linkedFramework("Carbon"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("UniformTypeIdentifiers")
            ]
        ),
        .testTarget(name: "ChimeIMETests", dependencies: ["ChimeIME"])
    ]
)
