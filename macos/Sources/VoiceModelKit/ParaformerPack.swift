import CryptoKit
import Foundation

/// Weights are data only. The executable and both native libraries stay in the signed app.
public struct ParaformerPack: Sendable {
    public struct Artifact: Sendable {
        let name: String
        let bytes: Int
        let sha256: String
    }

    static let revision = "8e40c43232a1c5c66c82111efc5820d3accca11b"
    static let manifest = [
        Artifact(name: "encoder.int8.onnx", bytes: 165462184, sha256: "81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a"),
        Artifact(name: "decoder.int8.onnx", bytes: 71664561, sha256: "f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f"),
        Artifact(name: "tokens.txt", bytes: 75756, sha256: "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6"),
    ]
    public let directory: URL
    let artifacts: [Artifact]

    public init(root: URL) {
        self.init(root: root, artifacts: Self.manifest)
    }

    init(root: URL, artifacts: [Artifact]) {
        directory = root.appendingPathComponent("paraformer-zh-en-int8-\(Self.revision)", isDirectory: true)
        self.artifacts = artifacts
    }

    public var installed: Bool {
        artifacts.allSatisfy { file in
            let info = try? directory.appendingPathComponent(file.name).resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            return info?.isRegularFile == true && info?.fileSize == file.bytes
        }
    }

    public func remove() throws {
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    public func download(status: @Sendable (String) async -> Void) async throws {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.httpShouldSetCookies = false
        cfg.urlCredentialStorage = nil
        cfg.timeoutIntervalForResource = 1800
        let session = URLSession(configuration: cfg)
        defer { session.invalidateAndCancel() }
        try await install(status: status) { file, destination in
            let base = URL(string: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/\(Self.revision)/")!
            let (tmp, response) = try await session.download(from: base.appendingPathComponent(file.name))
            defer { try? FileManager.default.removeItem(at: tmp) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }
            try FileManager.default.moveItem(at: tmp, to: destination)
        }
    }

    /// Stage on the same filesystem, verify each digest off the UI thread, publish once.
    func install(status: @Sendable (String) async -> Void,
                 fetch: @Sendable (Artifact, URL) async throws -> Void) async throws {
        guard !installed else { return }
        let fm = FileManager.default
        let staging = directory.deletingLastPathComponent().appendingPathComponent(".partial-\(UUID())")
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        for (index, file) in artifacts.enumerated() {
            try Task.checkCancellation()
            await status("下载并校验 \(index + 1)/\(artifacts.count)：\(file.name)")
            let dest = staging.appendingPathComponent(file.name)
            try await fetch(file, dest)
            let handle = try FileHandle(forReadingFrom: dest)
            defer { try? handle.close() }
            var digest = SHA256()
            var size = 0
            while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                try Task.checkCancellation()
                size += chunk.count
                digest.update(data: chunk)
            }
            guard size == file.bytes, digest.finalize().map({ String(format: "%02x", $0) }).joined() == file.sha256 else {
                throw NSError(domain: "ChimeVoice", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(file.name) 校验失败，请重新下载。"])
            }
        }
        try Task.checkCancellation()
        if installed { return }
        // Repair an incomplete install only after its replacement is fully verified.
        try remove()
        try fm.moveItem(at: staging, to: directory)
    }
}
