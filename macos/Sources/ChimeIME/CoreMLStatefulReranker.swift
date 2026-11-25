import CoreML
import Foundation

/// Optional macOS 15+ Apple Neural Engine (ANE) runner using CoreML `MLState` in-place stateful KV-cache
/// when a compiled `ChimeReranker.mlmodelc` or `.mlpackage` is installed in Application Support.
/// Gracefully no-ops on macOS 14 Sonoma where MLX Metal handles all Stage-2 reranking.
@MainActor
public final class CoreMLStatefulReranker {
    public static let shared = CoreMLStatefulReranker()

    private var model: MLModel?
    private var kvStateStorage: AnyObject?

    private init() {
        loadInstalledModelIfPresent()
    }

    public var isLoaded: Bool {
        model != nil && kvStateStorage != nil
    }

    public func loadInstalledModelIfPresent() {
        guard #available(macOS 15.0, *) else {
            return
        }
        guard let modelsDir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("Chime/models", isDirectory: true)
        else {
            return
        }

        let compiledURL = modelsDir.appendingPathComponent("ChimeReranker.mlmodelc")
        let packageURL = modelsDir.appendingPathComponent("ChimeReranker.mlpackage")

        let config = MLModelConfiguration()
        config.computeUnits = .all

        if FileManager.default.fileExists(atPath: compiledURL.path),
           let loaded = try? MLModel(contentsOf: compiledURL, configuration: config)
        {
            self.model = loaded
            self.kvStateStorage = loaded.makeState()
        } else if FileManager.default.fileExists(atPath: packageURL.path),
                  let freshCompiled = try? MLModel.compileModel(at: packageURL),
                  let loaded = try? MLModel(contentsOf: freshCompiled, configuration: config)
        {
            self.model = loaded
            self.kvStateStorage = loaded.makeState()
        }
    }

    /// Evaluates a single step using CoreML's macOS 15 `MLState` stateful KV-cache.
    public func stepLogits(inputTokenId: Int32) -> MLMultiArray? {
        guard #available(macOS 15.0, *),
              let model,
              let kvState = kvStateStorage as? MLState
        else {
            return nil
        }
        guard let inputArray = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
            return nil
        }
        inputArray[0] = NSNumber(value: inputTokenId)
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray]),
              let out = try? model.prediction(from: provider, using: kvState)
        else {
            return nil
        }
        return out.featureValue(for: "logits")?.multiArrayValue
    }
}
