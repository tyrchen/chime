import Combine
import Foundation
import VoiceModelKit

@MainActor
final class VoiceModelManager: ObservableObject {
    static let shared = VoiceModelManager()
    let pack: ParaformerPack
    @Published private(set) var installed: Bool
    @Published private(set) var progress: String?
    @Published private(set) var error: String?
    private var task: Task<Void, Never>?

    private init() {
        let support = URL.applicationSupportDirectory.appendingPathComponent("Chime/voice-models", isDirectory: true)
        let pack = ParaformerPack(root: support)
        self.pack = pack
        installed = pack.installed
    }

    func refresh() { installed = pack.installed }

    func download() {
        guard task == nil else { return }
        progress = "准备下载…"
        error = nil
        task = Task {
            do {
                try await pack.download { message in
                    await MainActor.run { self.progress = message }
                }
            } catch {
                if !Task.isCancelled { self.error = error.localizedDescription }
            }
            refresh()
            progress = nil
            task = nil
        }
    }

    func cancel() { task?.cancel() }

    func remove() {
        guard task == nil else { return }
        error = nil
        do { try pack.remove() } catch { self.error = error.localizedDescription }
        refresh()
    }
}
