import Foundation
import Testing
@testable import VoiceModelKit

private func withFixture(_ test: (ParaformerPack) async throws -> Void) async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let files = ["enc", "dec"].map {
        ParaformerPack.Artifact(name: $0, bytes: 3, sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
    try await test(ParaformerPack(root: root, artifacts: files))
}

@Test func verifiedInstallAndRemoval() async throws {
    try await withFixture { pack in
        #expect(!pack.installed)
        try await pack.install(status: { _ in }) { _, dest in
            #expect(!pack.installed)
            try Data("abc".utf8).write(to: dest)
        }
        #expect(pack.installed)
        try await pack.install(status: { _ in }) { _, _ in Issue.record("downloaded again") }
        try pack.remove()
        #expect(!pack.installed)
    }
}

@Test func corruptDownloadDoesNotPublish() async throws {
    try await withFixture { pack in
        await #expect(throws: (any Error).self) {
            try await pack.install(status: { _ in }) { file, dest in
                try Data((file.name == "dec" ? "bad" : "abc").utf8).write(to: dest)
            }
        }
        #expect(!pack.installed)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: pack.directory.deletingLastPathComponent().path)
        #expect(remaining.isEmpty)
    }
}

@Test func cancelledDownloadDoesNotPublish() async throws {
    try await withFixture { pack in
        let task = Task {
            try await pack.install(status: { _ in }) { _, dest in
                try Data("abc".utf8).write(to: dest)
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(!pack.installed)
    }
}
