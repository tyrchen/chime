import AppKit
import AVFoundation
import Carbon
import CChimeFFI
import Foundation
import InputMethodKit
import OSLog
import Security

/// Options sent to `chime_voice_start` / `chime_voice_test_endpoint` (mirrors Rust `VoiceSettings`).
struct VoiceRuntimeSettings: Encodable {
    let endpoint: String
    let model: String
    let language: String
    let prompt: String
    let live_preview: Bool
    let partial_interval_ms: UInt64
    let request_timeout_ms: UInt64
    let max_record_secs: UInt64
    let chinese_punctuation: Bool

    init(config: ChimeConfigDTO) {
        let v = config.voice
        endpoint = v.endpoint
        model = v.model
        language = v.language
        prompt = v.prompt
        live_preview = v.live_preview
        partial_interval_ms = v.partial_interval_ms
        request_timeout_ms = v.request_timeout_ms
        max_record_secs = v.max_record_secs
        chinese_punctuation = config.chinese_punctuation
    }

    func jsonString() -> String {
        guard let data = try? JSONEncoder().encode(self), let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }
}

/// Session state decoded from `chime_voice_snapshot`.
struct VoiceSnapshotDTO: Decodable {
    let state: String
    let text: String
    let level: Float
    let elapsed_ms: UInt64
    let busy: Bool
    let error: String?
    let limit_reached: Bool
}

/// Result of `chime_voice_test_endpoint`.
public struct VoiceEndpointProbe: Decodable, Sendable {
    public let ok: Bool
    public let text: String
    public let latency_ms: UInt64
    public let error: String?
}

/// Optional bearer token for the speech server, kept in the login Keychain instead of `config.toml`.
public enum ChimeVoiceKeychain {
    private static let service = "app.tyrchen.inputmethod.Chime.voice"
    private static let account = "asr-api-key"

    private static var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
    }

    public static func read() -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let key = String(data: data, encoding: .utf8),
              !key.isEmpty
        else {
            return nil
        }
        return key
    }

    /// Stores `key`; an empty string deletes the entry.
    @discardableResult
    public static func write(_ key: String) -> Bool {
        SecItemDelete(baseQuery as CFDictionary)
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return true
        }
        var add = baseQuery
        add[kSecValueData as String] = Data(trimmed.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }
}

/// Feeds one tap buffer to `AVAudioConverter` exactly once per `convert` call.
private final class OneShotFeeder: @unchecked Sendable {
    private var buffer: AVAudioPCMBuffer?

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        guard let buffer else {
            // `.noDataNow` (not `.endOfStream`) keeps the resampler state for the next buffer.
            status.pointee = .noDataNow
            return nil
        }
        self.buffer = nil
        status.pointee = .haveData
        return buffer
    }
}

/// Resamples microphone buffers to 16 kHz mono Float32 and pushes them to Rust. Runs only on the
/// `AVAudioEngine` tap thread; the Rust voice handle is internally synchronized.
private final class VoicePCMSink: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let outFormat: AVAudioFormat
    private let ratio: Double
    private let handle: OpaquePointer

    init?(inputFormat: AVAudioFormat, handle: OpaquePointer) {
        guard let out = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: ChimeVoiceController.sampleRate,
            channels: 1,
            interleaved: false
        ),
            let conv = AVAudioConverter(from: inputFormat, to: out)
        else {
            return nil
        }
        converter = conv
        outFormat = out
        ratio = out.sampleRate / inputFormat.sampleRate
        self.handle = handle
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        let feeder = OneShotFeeder(buffer)
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, outStatus in
            feeder.next(outStatus)
        }
        guard status != .error, out.frameLength > 0, let channel = out.floatChannelData?[0] else { return }
        chime_voice_push_pcm(handle, channel, Int(out.frameLength))
    }

    /// Built outside any actor so the tap block is not `@MainActor`-isolated (it runs on the
    /// realtime audio thread).
    nonisolated static func makeTapBlock(_ sink: VoicePCMSink) -> AVAudioNodeTapBlock {
        { buffer, _ in sink.consume(buffer) }
    }
}

enum VoiceCaptureError: LocalizedError {
    case noInputDevice
    case unsupportedFormat

    var errorDescription: String? {
        switch self {
        case .noInputDevice: "没有可用的麦克风输入设备"
        case .unsupportedFormat: "麦克风音频格式无法转换为 16 kHz 单声道"
        }
    }
}

/// Owns one `AVAudioEngine` per recording so device changes (AirPods, USB mics) are picked up.
@MainActor
private final class VoiceAudioCapture {
    private var engine: AVAudioEngine?

    func start(handle: OpaquePointer) throws {
        stop()
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw VoiceCaptureError.noInputDevice
        }
        guard let sink = VoicePCMSink(inputFormat: format, handle: handle) else {
            throw VoiceCaptureError.unsupportedFormat
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format, block: VoicePCMSink.makeTapBlock(sink))
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        self.engine = engine
    }

    func stop() {
        guard let engine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        self.engine = nil
    }
}

/// Push-to-talk dictation: hold the configured modifier, speak, release to commit.
///
/// While recording, the live transcript is shown in `VoiceInputPanel` and inline as marked text in
/// the focused client; release commits the final text, `Esc` cancels.
@MainActor
public final class ChimeVoiceController {
    public static let shared = ChimeVoiceController()
    nonisolated static let sampleRate: Double = 16_000

    private enum Phase {
        case idle
        case arming
        case recording
        case finishing
        case formatting
        case ranking
        case reviewing
    }

    private struct HotkeySpec {
        let keyCode: UInt16
        let flag: NSEvent.ModifierFlags
        /// `NX_DEVICE*KEYMASK` bit identifying the left/right side (`0` = not side-specific).
        let deviceMask: UInt
        let label: String
    }

    /// Hold this long before recording starts, so `⌥`-chords and quick taps keep working.
    private static let holdThreshold: Duration = .milliseconds(280)
    private static let allDeviceMasks: UInt = 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x2000
    private static let chordModifiers: NSEvent.ModifierFlags = [.shift, .control, .option, .command, .function]
    private static let escKeyCode: UInt16 = 0x35

    private let handle: OpaquePointer?
    private let capture = VoiceAudioCapture()
    private let decoder = JSONDecoder()
    private var phase: Phase = .idle
    private var hotkeyDown = false
    private var activeSpec: HotkeySpec?
    private var armTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var formatTask: Task<Void, Never>?
    private var reviewTimeout: Task<Void, Never>?
    private var alternative: String?
    private var neuralAlternative = false
    private static let reviewLog = Logger(subsystem: "app.tyrchen.inputmethod.Chime", category: "VoiceReview")
    private var utteranceStart: Int?
    private var inlinePreview: String?
    private var reviewSite: VoiceReviewSite?
    private let reviewTabTap = VoiceReviewTabTap()
    private var reviewID: UUID?
    private var reviewHoldTask: Task<Void, Never>?
    private var autoPunctuation = true
    private var tidyDictation = false
    private var target: (key: ObjectIdentifier, client: IMKTextInput)?
    private var hasMarkedText = false
    private var lastText = ""

    private init() {
        handle = chime_voice_new()
    }

    private static func spec(for hotkey: String) -> HotkeySpec {
        switch hotkey {
        case "right_command":
            HotkeySpec(keyCode: 0x36, flag: .command, deviceMask: 0x10, label: "右 ⌘")
        case "right_control":
            HotkeySpec(keyCode: 0x3E, flag: .control, deviceMask: 0x2000, label: "右 ⌃")
        case "fn":
            HotkeySpec(keyCode: 0x3F, flag: .function, deviceMask: 0, label: "Fn")
        default:
            HotkeySpec(keyCode: 0x3D, flag: .option, deviceMask: 0x40, label: "右 ⌥")
        }
    }

    public static func hotkeyLabel(_ hotkey: String) -> String {
        spec(for: hotkey).label
    }

    // MARK: - Event intake

    /// Called for every `.flagsChanged` from both `IMKInputController.handle` and the global
    /// monitor. Idempotent: duplicate deliveries of the same flags are harmless.
    public func handleFlagsChanged(keyCode: UInt16, flags: NSEvent.ModifierFlags) {
        let voice = ChimeEngineBridge.shared.currentConfig.voice
        guard voice.enabled else {
            cancelArming()
            hotkeyDown = false
            return
        }
        let spec = Self.spec(for: voice.hotkey)
        let raw = flags.rawValue
        let down: Bool = if spec.deviceMask != 0 && (raw & Self.allDeviceMasks) != 0 {
            (raw & spec.deviceMask) != 0
        } else if keyCode == spec.keyCode {
            flags.contains(spec.flag)
        } else if !flags.contains(spec.flag) {
            false
        } else {
            hotkeyDown
        }
        let otherHeld = !flags.intersection(Self.chordModifiers.subtracting(spec.flag)).isEmpty

        switch phase {
        case .idle:
            if down && !hotkeyDown && !otherHeld {
                beginArming(spec)
            }
        case .arming:
            if !down || otherHeld {
                cancelArming()
            }
        case .recording:
            if !down {
                finishRecording()
            }
        case .reviewing:
            // A fresh short tap confirms without sending Tab. A hold still starts the next
            // dictation; neither the release that ended recording nor a chord confirms.
            if otherHeld {
                reviewHoldTask?.cancel(); reviewHoldTask = nil
            } else if down && !hotkeyDown {
                reviewHoldTask = Task { @MainActor [weak self] in
                    try? await Task.sleep(for: Self.holdThreshold)
                    guard let self, !Task.isCancelled, self.phase == .reviewing else { return }
                    let held = NSEvent.modifierFlags
                    guard held.contains(spec.flag), held.intersection(Self.chordModifiers.subtracting(spec.flag)).isEmpty,
                          NSEvent.pressedMouseButtons == 0 else { return }
                    self.reviewHoldTask = nil
                    if self.resolveSuggestion(accept: false) { self.startRecording(spec) }
                }
            } else if !down && hotkeyDown && reviewHoldTask != nil {
                resolveSuggestion(accept: true)
            }
        case .finishing, .formatting, .ranking:
            break
        }
        hotkeyDown = down
    }

    /// Returns `true` when the key was consumed by dictation.
    public func interceptKeyDown(keyCode: UInt16, flags: NSEvent.ModifierFlags) -> Bool {
        switch phase {
        case .idle:
            return false
        case .arming:
            // A real chord such as `⌥ + key`: not a dictation request.
            cancelArming()
            return false
        case .recording:
            if keyCode == Self.escKeyCode {
                cancelSession()
            }
            return true
        case .ranking:
            // A real key takes priority. No invisible candidate can be adopted.
            if keyCode == Self.escKeyCode && flags.intersection(Self.chordModifiers).isEmpty {
                resolveSuggestion(accept: false)
                return true
            }
            resolveSuggestion(accept: false)
            return false
        case .reviewing:
            let plain = flags.intersection(Self.chordModifiers).isEmpty
            let keep = plain && keyCode == Self.escKeyCode
            resolveSuggestion(accept: false)
            return keep
        case .finishing, .formatting:
            if keyCode == Self.escKeyCode {
                cancelSession()
                return true
            }
            guard !IsSecureEventInputEnabled(), ChimeEngineBridge.shared.currentConfig.voice.enabled else {
                cancelSession()
                return false
            }
            // Typing takes priority over the optional, asynchronous formatting pass.
            let latest = handle.flatMap { snapshot($0) }
            endSession(commit: latest?.state == "done" ? latest?.text : lastText, error: latest?.error)
            return false
        }
    }

    public func noteMouseDown() {
        cancelArming()
    }

    public func controllerDeactivated(_ key: ObjectIdentifier) {
        cancelArming()
        if let target, target.key == key, phase != .idle {
            cancelSession()
        }
    }

    // MARK: - Session lifecycle

    private func beginArming(_ spec: HotkeySpec) {
        guard ChimeEngineBridge.isChimeSelectedInTIS(),
              ChimeEngineBridge.shared.activeTextClient() != nil,
              !IsSecureEventInputEnabled()
        else {
            return
        }
        phase = .arming
        armTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: Self.holdThreshold)
            guard let self, !Task.isCancelled, self.phase == .arming else { return }
            self.armTask = nil
            // Re-check the live hardware state: an `IMK` client may not forward every event.
            let held = NSEvent.modifierFlags
            let clean = held.intersection(Self.chordModifiers.subtracting(spec.flag)).isEmpty
            guard held.contains(spec.flag), clean, NSEvent.pressedMouseButtons == 0 else {
                self.phase = .idle
                if !held.contains(spec.flag) {
                    self.hotkeyDown = false
                }
                return
            }
            self.startRecording(spec)
        }
    }

    private func cancelArming() {
        armTask?.cancel()
        armTask = nil
        if phase == .arming {
            phase = .idle
        }
    }

    private func startRecording(_ spec: HotkeySpec) {
        guard !IsSecureEventInputEnabled(), ChimeEngineBridge.shared.currentConfig.voice.enabled,
              let handle, let active = ChimeEngineBridge.shared.activeTextClient() else {
            phase = .idle
            return
        }
        let caret = ChimeEngineBridge.shared.caretRect(for: active.key)
        let config = ChimeEngineBridge.shared.currentConfig
        if config.voice.backend != "http", !VoiceModelManager.shared.pack.installed {
            phase = .idle
            VoiceInputPanel.shared.flash(message: "请先在「偏好设置 → 语音输入」中下载 Paraformer 模型", anchor: caret, isError: true)
            return
        }

        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            break
        case .notDetermined:
            phase = .idle
            VoiceInputPanel.shared.flash(
                message: "首次使用需要麦克风权限，授权后再次按住 \(spec.label) 说话",
                anchor: caret,
                isError: false
            )
            AVCaptureDevice.requestAccess(for: .audio) { _ in }
            return
        default:
            phase = .idle
            VoiceInputPanel.shared.flash(
                message: "麦克风权限已关闭：请在「系统设置 › 隐私与安全性 › 麦克风」中允许 Chime",
                anchor: caret,
                isError: true
            )
            return
        }

        let errPtr: UnsafeMutablePointer<CChar>?
        if config.voice.backend == "http" {
            let settingsJSON = VoiceRuntimeSettings(config: config).jsonString()
            let apiKey = ChimeVoiceKeychain.read()
            let rate = UInt32(Self.sampleRate)
            errPtr = settingsJSON.withCString { ptr in
                if let apiKey { apiKey.withCString { chime_voice_start(handle, ptr, $0, rate) } }
                else { chime_voice_start(handle, ptr, nil, rate) }
            }
        } else {
            let contents = Bundle.main.bundleURL.appendingPathComponent("Contents")
            errPtr = contents.path.withCString { appPath in
                VoiceModelManager.shared.pack.directory.path.withCString { modelPath in
                    config.voice.sense_final_dir.withCString { finalDir in
                        chime_voice_start_local(handle, appPath, modelPath, finalDir, config.voice.auto_punctuation, config.voice.tidy_dictation,
                                                config.voice.live_preview, config.voice.max_record_secs)
                    }
                }
            }
        }
        if let errPtr {
            let message = String(cString: errPtr)
            chime_string_free(errPtr)
            phase = .idle
            VoiceInputPanel.shared.flash(message: "语音输入设置有误：\(message)", anchor: caret, isError: true)
            return
        }

        ChimeEngineBridge.shared.discardCompositionForVoice(controllerKey: active.key)
        CandidatePanel.shared.hide()
        StatusToastPanel.shared.hide()
        do {
            try capture.start(handle: handle)
        } catch {
            chime_voice_cancel(handle)
            phase = .idle
            VoiceInputPanel.shared.flash(
                message: "无法打开麦克风：\(error.localizedDescription)",
                anchor: caret,
                isError: true
            )
            return
        }

        autoPunctuation = config.voice.auto_punctuation
        tidyDictation = config.voice.tidy_dictation
        target = active
        let selection = active.client.selectedRange()
        utteranceStart = selection.location == NSNotFound ? nil : selection.location
        activeSpec = spec
        hasMarkedText = false
        lastText = ""
        phase = .recording
        VoiceInputPanel.shared.present(anchor: caret, hint: recordingHint(spec: spec, elapsedMs: 0))
        startPolling()
    }

    private func finishRecording() {
        guard phase == .recording else { return }
        capture.stop()
        if let handle {
            chime_voice_stop(handle)
        }
        phase = .finishing
    }

    private func cancelSession() {
        capture.stop()
        if let handle {
            chime_voice_cancel(handle)
        }
        endSession(commit: nil, error: nil)
    }

    /// Commits `text` (if any) to the target client and returns to idle.
    private func endSession(commit text: String?, error: String?, replacement: NSRange = NSRange(location: NSNotFound, length: 0)) {
        reviewID = nil
        reviewTabTap.endReview()
        reviewHoldTask?.cancel(); reviewHoldTask = nil
        reviewSite = nil
        inlinePreview = nil
        utteranceStart = nil
        reviewTimeout?.cancel()
        reviewTimeout = nil
        alternative = nil
        neuralAlternative = false
        formatTask?.cancel()
        formatTask = nil
        pollTask?.cancel()
        pollTask = nil
        capture.stop()
        if let handle {
            chime_voice_end(handle)
        }
        let client = target?.client
        target = nil
        activeSpec = nil
        phase = .idle

        let committed = text ?? ""
        if let client {
            if !committed.isEmpty {
                client.insertText(committed, replacementRange: replacement)
            } else if hasMarkedText {
                client.setMarkedText(
                    "",
                    selectionRange: NSRange(location: 0, length: 0),
                    replacementRange: NSRange(location: NSNotFound, length: 0)
                )
            }
        }
        hasMarkedText = false
        lastText = ""

        if let error {
            VoiceInputPanel.shared.flash(message: "语音识别出错：\(error)", anchor: nil, isError: true)
        } else if text != nil && committed.isEmpty {
            VoiceInputPanel.shared.flash(message: "没有识别到语音", anchor: nil, isError: false)
        } else {
            VoiceInputPanel.shared.dismiss()
        }
    }

    // MARK: - Live updates

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(60))
                guard let self, !Task.isCancelled else { return }
                if !self.tick() {
                    return
                }
            }
        }
    }

    /// Applies one snapshot; returns `false` once the session has ended.
    private func tick() -> Bool {
        guard !IsSecureEventInputEnabled(), ChimeEngineBridge.shared.currentConfig.voice.enabled else {
            cancelSession()
            return false
        }
        guard let handle, let snap = snapshot(handle) else {
            endSession(commit: nil, error: nil)
            return false
        }
        // Safety net for a missed key-up: the modifier is physically released.
        if phase == .recording, let spec = activeSpec, !NSEvent.modifierFlags.contains(spec.flag) {
            hotkeyDown = false
            finishRecording()
        }
        // `max_record_secs` reached: Rust already stopped accepting audio.
        if phase == .recording && snap.state == "finishing" {
            capture.stop()
            phase = .finishing
        }
        switch snap.state {
        case "done":
            completeRecognition(snap)
            return false
        case "cancelled":
            endSession(commit: nil, error: nil)
            return false
        default:
            break
        }

        if snap.text != lastText {
            lastText = snap.text
            showInline(snap.text)
        }
        let hint: String = if let error = snap.error {
            "⚠︎ \(error)"
        } else if phase == .recording, let spec = activeSpec {
            recordingHint(spec: spec, elapsedMs: snap.elapsed_ms)
        } else {
            "识别中… · Esc 取消"
        }
        VoiceInputPanel.shared.update(
            text: snap.text,
            level: snap.level,
            hint: hint,
            isError: snap.error != nil,
            listening: phase == .recording
        )
        return true
    }

    private func completeRecognition(_ snap: VoiceSnapshotDTO) {
        let raw = snap.text
        guard snap.error == nil, !raw.isEmpty else {
            Self.reviewLog.info("review skipped: ASR error or empty final")
            endSession(commit: raw, error: snap.error)
            return
        }
        capture.stop()
        if let handle { chime_voice_end(handle) }
        guard autoPunctuation || tidyDictation else {
            offerSuggestion(for: raw)
            return
        }
        phase = .formatting
        lastText = raw // if the user types immediately, prefer the full ASR final to a stale preview
        VoiceInputPanel.shared.update(text: raw, level: 0, hint: "整理文字中… · Esc 取消", isError: false, listening: false)
        let contents = Bundle.main.bundleURL.appendingPathComponent("Contents").path
        let punct = autoPunctuation, tidy = tidyDictation
        formatTask = Task { @MainActor [weak self] in
            let text = await Task.detached(priority: .userInitiated) {
                contents.withCString { path in raw.withCString { input -> String in
                    guard let ptr = chime_voice_format_final(path, input, punct, tidy) else { return raw }
                    defer { chime_string_free(ptr) }
                    return String(cString: ptr)
                } }
            }.value
            guard let self, !Task.isCancelled, self.phase == .formatting else { return }
            guard !IsSecureEventInputEnabled(), ChimeEngineBridge.shared.currentConfig.voice.enabled,
                  let active = ChimeEngineBridge.shared.activeTextClient(), active.key == self.target?.key else {
                self.cancelSession()
                return
            }
            self.offerSuggestion(for: text)
        }
    }

    private func offerSuggestion(for text: String) {
        let bridge = ChimeEngineBridge.shared
        guard bridge.currentConfig.voice.lexicon_suggestions else {
            Self.reviewLog.info("review disabled")
            endSession(commit: text, error: nil)
            return
        }
        if let candidate = bridge.voiceLexiconSuggestion(text) {
            presentSuggestion(candidate, original: text, neural: false)
            return
        }
        guard bridge.currentConfig.voice.pinyin_rerank, let job = bridge.startVoiceReview(text) else {
            Self.reviewLog.info("no dictionary suggestion or neural job")
            endSession(commit: text, error: nil)
            return
        }
        phase = .ranking
        guard prepareReview(text) else {
            chime_voice_review_free(job)
            cancelSession()
            return
        }
        VoiceInputPanel.shared.update(text: text, level: 0, hint: "比较同音候选…", isError: false, listening: false)
        formatTask = Task { @MainActor [weak self] in
            defer { chime_voice_review_free(job) }
            while !Task.isCancelled, let self, self.phase == .ranking {
                var ptr: UnsafeMutablePointer<CChar>?
                if chime_voice_review_poll(job, &ptr) {
                    defer { if let ptr { chime_string_free(ptr) } }
                    guard self.reviewTargetIsCurrent(),
                          bridge.currentConfig.voice.lexicon_suggestions, bridge.currentConfig.voice.pinyin_rerank,
                          let ptr else { self.resolveSuggestion(accept: false); return }
                    self.presentSuggestion(String(cString: ptr), original: text, neural: true)
                    return
                }
                try? await Task.sleep(for: .milliseconds(12))
            }
        }
    }

    private func presentSuggestion(_ candidate: String, original text: String, neural: Bool) {
        guard prepareReview(text) else { cancelSession(); return }
        phase = .reviewing
        alternative = candidate
        neuralAlternative = neural
        let label = neural ? "同音建议" : "词库建议"
        Self.reviewLog.info("show suggestion HUD; copy-only: \(self.reviewSite == .copyOnly, privacy: .public)")
        let id = UUID()
        reviewID = id
        let confirm: () -> Void = { [weak self] in
            guard let self, self.phase == .reviewing, self.reviewID == id else { return }
            self.resolveSuggestion(accept: true)
        }
        let action = reviewSite == .copyOnly ? "复制" : "采用"
        let fallbackHint = "\(label) · 轻点\(activeSpec?.label ?? "语音键")\(action) · Esc 保留原文"
        let showFallback = { [weak self] in
            guard self?.reviewID == id else { return }
            VoiceInputPanel.shared.showReview(text: candidate, hint: fallbackHint, button: action, adopt: confirm)
        }
        if reviewTabTap.begin(bundleID: target?.client.bundleIdentifier(), adopt: confirm, unavailable: showFallback) {
            VoiceInputPanel.shared.showReview(text: candidate, hint: "\(label) · Tab \(action) · Esc 保留原文", button: action, adopt: confirm)
        } else {
            showFallback()
        }
        reviewTimeout = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(4))
            guard !Task.isCancelled, self?.phase == .reviewing else { return }
            self?.resolveSuggestion(accept: false)
        }
    }

    private func site(for text: String, client: IMKTextInput) -> VoiceReviewSite {
        VoiceReviewSite.probe(text: text, start: utteranceStart, marked: client.markedRange(), selected: client.selectedRange()) {
            client.attributedSubstring(from: $0)?.string
        }
    }

    private func prepareReview(_ text: String) -> Bool {
        guard reviewTargetIsCurrent(), let client = target?.client else {
            Self.reviewLog.info("review skipped: focus or secure-input guard")
            return false
        }
        if reviewSite != nil { return true } // ranking -> HUD, don't write twice
        lastText = text
        if inlinePreview == nil { showInline(text) }
        var current = site(for: text, client: client)
        if current == .copyOnly, let inlinePreview {
            switch site(for: inlinePreview, client: client) {
            case .marked:
                showInline(text)
                current = site(for: text, client: client)
            case .committed(let range):
                // Host already committed the partial. Update only that verified span, once.
                client.insertText(text, replacementRange: range)
                current = site(for: text, client: client)
                hasMarkedText = false
                reviewSite = current
                return true
            case .copyOnly: break
            }
        }
        if current == .copyOnly {
            // Preserve the old normal dictation commit, but *still* offer the suggestion.
            // If we can verify what was inserted, confirmation may replace that exact range.
            client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: 0))
            current = site(for: text, client: client)
            if case .marked = current { current = .copyOnly }
        }
        if case .marked = current { hasMarkedText = true } else { hasMarkedText = false }
        reviewSite = current
        return true
    }

    private func reviewTargetIsCurrent() -> Bool {
        let bridge = ChimeEngineBridge.shared
        guard !IsSecureEventInputEnabled(), bridge.currentConfig.voice.enabled,
              let active = bridge.activeTextClient(), let target else { return false }
        return active.key == target.key && (active.client as AnyObject) === (target.client as AnyObject)
    }

    @discardableResult
    private func resolveSuggestion(accept: Bool) -> Bool {
        let config = ChimeEngineBridge.shared.currentConfig.voice
        guard reviewTargetIsCurrent(), let target else {
            cancelSession()
            return false
        }
        let adopt = accept && config.lexicon_suggestions && (!neuralAlternative || config.pinyin_rerank)
        if reviewSite == .copyOnly {
            if adopt, let alternative {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(alternative, forType: .string)
            }
            endSession(commit: nil, error: nil) // fallback has already committed the original
            return true
        }
        let current = site(for: lastText, client: target.client)
        switch (reviewSite, current) {
        case (.marked(let old), .marked(let now)) where old == now:
            endSession(commit: adopt ? alternative : lastText, error: nil)
        case (.marked(let old), .committed(let now)) where old == now,
             (.committed(let old), .committed(let now)) where old == now:
            hasMarkedText = false
            endSession(commit: adopt ? alternative : nil, error: nil, replacement: now)
        default:
            self.target = nil // caret moved or text changed; never clear another composition
            endSession(commit: nil, error: nil)
            return false
        }
        return true
    }

    private func recordingHint(spec: HotkeySpec, elapsedMs: UInt64) -> String {
        let secs = Int(elapsedMs / 1000)
        return String(format: "%ld:%02ld", secs / 60, secs % 60) + " · 松开 \(spec.label) 上屏 · Esc 取消"
    }

    private func showInline(_ text: String) {
        guard let client = target?.client else { return }
        inlinePreview = text.isEmpty ? nil : text
        if text.isEmpty {
            guard hasMarkedText else { return }
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )
            hasMarkedText = false
            return
        }
        client.setMarkedText(
            text,
            selectionRange: NSRange(location: (text as NSString).length, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        hasMarkedText = true
    }

    private func snapshot(_ handle: OpaquePointer) -> VoiceSnapshotDTO? {
        guard let raw = chime_voice_snapshot(handle) else { return nil }
        defer { chime_string_free(raw) }
        let data = Data(String(cString: raw).utf8)
        return try? decoder.decode(VoiceSnapshotDTO.self, from: data)
    }

    // MARK: - Settings helpers

    /// Sends a short silent clip to the configured server (off the main thread).
    public static func probeEndpoint(config: ChimeConfigDTO, apiKey: String?) async -> VoiceEndpointProbe {
        let json = VoiceRuntimeSettings(config: config).jsonString()
        return await Task.detached(priority: .userInitiated) {
            let raw: UnsafeMutablePointer<CChar>? = json.withCString { settingsPtr in
                if let apiKey, !apiKey.isEmpty {
                    apiKey.withCString { keyPtr in chime_voice_test_endpoint(settingsPtr, keyPtr) }
                } else {
                    chime_voice_test_endpoint(settingsPtr, nil)
                }
            }
            guard let raw else {
                return VoiceEndpointProbe(ok: false, text: "", latency_ms: 0, error: "内部错误")
            }
            defer { chime_string_free(raw) }
            let data = Data(String(cString: raw).utf8)
            return (try? JSONDecoder().decode(VoiceEndpointProbe.self, from: data))
                ?? VoiceEndpointProbe(ok: false, text: "", latency_ms: 0, error: "无法解析测试结果")
        }.value
    }
}
