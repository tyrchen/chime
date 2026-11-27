import AppKit
import CryptoKit
import Foundation

/// Signed static release manifest published at `https://tyrchen.github.io/chime/releases-manifest.json`.
public struct ChimeReleaseManifest: Codable, Sendable, Equatable {
    public let version: String
    public let pkg_sha256: String
    public let zip_sha256: String
    public let classics_sha256: String
    public let tech_sha256: String
}

/// Metadata for official first-party `.imepack` cartridges available for 1-click installation.
public struct OfficialPackCatalogItem: Identifiable, Sendable {
    public let id: String
    public let title: String
    public let routingBadge: String
    public let summary: String
    public let approxSize: String

    public static let catalog: [OfficialPackCatalogItem] = [
        OfficialPackCatalogItem(
            id: "classics-poetry",
            title: "古典诗词曲赋与成语全库",
            routingBadge: "古汉语语境自动激活",
            summary: "26.6 万古典词条 + 23.1 万条全篇逐句 Tab 续写链 + 18L Rank-8 古文 LoRA",
            approxSize: "~24 MB"
        ),
        OfficialPackCatalogItem(
            id: "tech-developer",
            title: "开发者中英技术双语包",
            routingBadge: "终端 / IDE 自动激活",
            summary: "Rust / Swift / AI / 云原生双语术语与精确驼峰大小写 + 18L Rank-8 技术 LoRA",
            approxSize: "~8.9 MB"
        )
    ]
}

/// Zero-telemetry, cryptographically verified (`SHA-256` + Apple `Developer ID` signature)
/// native updater and `.imepack` cartridge downloader for Chime IME.
@MainActor
public final class ChimeUpdateManager: ObservableObject {
    public static let shared = ChimeUpdateManager()

    private nonisolated static let expectedTeamId = "P5YL2MWHJ8"
    private nonisolated static let expectedBundleId = "app.tyrchen.inputmethod.Chime"
    private static let manifestURLs: [URL] = [
        URL(string: "https://tyrchen.github.io/chime/releases-manifest.json")!,
        URL(string: "https://raw.githubusercontent.com/tyrchen/chime/main/site/releases-manifest.json")!
    ]
    private static let checkIntervalSeconds: TimeInterval = 86_400
    private static let lastAutoCheckDefaultsKey = "chime.updater.lastAutoCheckEpoch"
    private static let lastPromptedVersionDefaultsKey = "chime.updater.lastPromptedVersion"

    @Published public private(set) var currentVersion: String
    @Published public private(set) var latestManifest: ChimeReleaseManifest?
    @Published public private(set) var availableAppUpdate: ChimeReleaseManifest?
    @Published public private(set) var isChecking: Bool = false
    @Published public private(set) var isUpdatingApp: Bool = false
    @Published public private(set) var activePackDownloadId: String?
    @Published public private(set) var statusMessage: String = ""

    private var periodicTimer: Timer?

    private init() {
        self.currentVersion = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.9.0"
    }

    /// Starts the non-intrusive 24-hour background update check scheduler (respects `auto_check_updates`).
    public func startBackgroundScheduler() {
        guard periodicTimer == nil else { return }

        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(8))
            await self?.runScheduledCheckIfDue()
        }

        periodicTimer = Timer.scheduledTimer(withTimeInterval: 21_600, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.runScheduledCheckIfDue()
            }
        }
    }

    /// Triggered when the user clicks "检查更新…" in the menu bar or Preferences.
    public func checkForUpdatesInteractive(openSettingsWindow: Bool = false) {
        if openSettingsWindow {
            ChimeSettingsWindowController.shared.showWindow()
        }
        Task { @MainActor in
            await refreshManifest(userInitiated: true)
        }
    }

    /// Fetches `releases-manifest.json` silently if not already cached in memory (used when opening
    /// the `.imepack` cartridge tab so version badges populate cleanly).
    public func ensureManifestLoadedQuietly() {
        guard latestManifest == nil, !isChecking else { return }
        let autoEnabled = ChimeEngineBridge.shared.getConfig()?.auto_check_updates ?? true
        guard autoEnabled else { return }
        Task { @MainActor in
            await refreshManifest(userInitiated: false)
        }
    }

    private func runScheduledCheckIfDue() async {
        let cfg = ChimeEngineBridge.shared.getConfig() ?? .default
        guard cfg.auto_check_updates else { return }

        let now = Date().timeIntervalSince1970
        let last = UserDefaults.standard.double(forKey: Self.lastAutoCheckDefaultsKey)
        guard now - last >= Self.checkIntervalSeconds else { return }

        UserDefaults.standard.set(now, forKey: Self.lastAutoCheckDefaultsKey)
        await refreshManifest(userInitiated: false)
    }

    /// Performs a zero-cookie, ephemeral read-only HTTPS GET of `releases-manifest.json`.
    public func refreshManifest(userInitiated: Bool) async {
        guard !isChecking else { return }
        isChecking = true
        if userInitiated {
            statusMessage = "正在检查最新版本…"
        }
        defer { isChecking = false }

        let session = Self.makeZeroTelemetrySession()
        var fetched: ChimeReleaseManifest?
        for url in Self.manifestURLs {
            do {
                let (data, resp) = try await session.data(from: url)
                guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    continue
                }
                let candidate = try JSONDecoder().decode(ChimeReleaseManifest.self, from: data)
                if let prev = fetched {
                    if Self.isVersionNewer(candidate.version, than: prev.version) {
                        fetched = candidate
                    }
                } else {
                    fetched = candidate
                }
            } catch {
                continue
            }
        }

        guard let manifest = fetched else {
            if userInitiated {
                statusMessage = "无法连接更新源，请检查网络或稍后重试。"
            }
            return
        }

        latestManifest = manifest
        UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: Self.lastAutoCheckDefaultsKey)

        if Self.isVersionNewer(manifest.version, than: currentVersion) {
            availableAppUpdate = manifest
            statusMessage = "发现新版本 v\(manifest.version)（当前 v\(currentVersion)），可一键安全升级。"
            if !userInitiated {
                maybePromptBackgroundUpdateOnce(manifest: manifest)
            }
        } else {
            availableAppUpdate = nil
            if userInitiated {
                statusMessage = "当前已是最新版本（v\(currentVersion)）。"
            }
            // Automatically hot-update any installed official `.imepack` cartridge whose version
            // lags the active app release (e.g. migrating older LoRAs to the 28L/36K base).
            await autoUpdateOutdatedOfficialPacks(manifest: manifest)
        }
    }

    private func autoUpdateOutdatedOfficialPacks(manifest: ChimeReleaseManifest) async {
        guard activePackDownloadId == nil else { return }
        guard let installed = ChimeEngineBridge.shared.getStatus()?.pack_details else { return }
        let officialIds: Set<String> = ["classics-poetry", "tech-developer"]
        for pack in installed where officialIds.contains(pack.id) {
            if Self.isVersionNewer(manifest.version, than: pack.version) {
                await downloadAndInstallOfficialPack(packId: pack.id, manifest: manifest)
            }
        }
    }

    private func maybePromptBackgroundUpdateOnce(manifest: ChimeReleaseManifest) {
        let lastPrompted = UserDefaults.standard.string(forKey: Self.lastPromptedVersionDefaultsKey)
        guard lastPrompted != manifest.version else { return }
        // Never interrupt active Pinyin composition!
        guard !CandidatePanel.shared.isVisible else { return }

        UserDefaults.standard.set(manifest.version, forKey: Self.lastPromptedVersionDefaultsKey)
        let alert = NSAlert()
        alert.messageText = "发现 Chime 拼音新版本 v\(manifest.version)"
        alert.informativeText = "当前版本为 v\(currentVersion)。新版本已通过 Apple 官方公证与 SHA-256 完整性签名校验，是否前往偏好设置一键升级？"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "立即升级")
        alert.addButton(withTitle: "稍后提醒")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            ChimeSettingsWindowController.shared.showWindow()
            installAvailableAppUpdate()
        }
    }

    /// Downloads, verifies (`SHA-256` + Apple Developer ID `P5YL2MWHJ8`), and installs the latest
    /// Chime release via either user-space `.zip` hot-swap (`~/Library/Input Methods`) or
    /// notarized `.pkg` upgrade (`/Library/Input Methods`).
    public func installAvailableAppUpdate() {
        guard let manifest = availableAppUpdate, !isUpdatingApp else { return }
        isUpdatingApp = true

        Task { @MainActor in
            defer { isUpdatingApp = false }
            let bundleURL = Bundle.main.bundleURL
            let parentDir = bundleURL.deletingLastPathComponent().path
            let isUserWritableInstall = FileManager.default.isWritableFile(atPath: parentDir)

            if isUserWritableInstall {
                await performUserSpaceZipUpdate(manifest: manifest, targetBundleURL: bundleURL)
            } else {
                await performSystemPkgUpdate(manifest: manifest)
            }
        }
    }

    private func performUserSpaceZipUpdate(manifest: ChimeReleaseManifest, targetBundleURL: URL) async {
        let ver = manifest.version
        guard let url = URL(string: "https://github.com/tyrchen/chime/releases/download/v\(ver)/Chime-v\(ver)-macos-arm64.zip") else {
            statusMessage = "无效的下载地址。"
            return
        }
        statusMessage = "正在下载 Chime v\(ver) 应用包…"
        let session = Self.makeZeroTelemetrySession()
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chime-update-\(UUID().uuidString)", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
            let (tmpFile, resp) = try await session.download(from: url)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "下载失败（HTTP 状态异常）。"
                return
            }

            let zipPath = workDir.appendingPathComponent("Chime-v\(ver).zip")
            try FileManager.default.moveItem(at: tmpFile, to: zipPath)

            statusMessage = "正在校验 SHA-256 与 Apple Developer ID 签名…"
            let actualHash = try Self.sha256Hex(of: zipPath)
            guard actualHash == manifest.zip_sha256.lowercased() else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "安全拦截：SHA-256 校验不匹配，已中止更新。"
                return
            }

            let unpackDir = workDir.appendingPathComponent("unpacked", isDirectory: true)
            try FileManager.default.createDirectory(at: unpackDir, withIntermediateDirectories: true)
            guard Self.runProcess("/usr/bin/ditto", ["-x", "-k", zipPath.path, unpackDir.path]).exitCode == 0 else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "解压应用包失败。"
                return
            }

            let newAppURL = unpackDir.appendingPathComponent("Chime.app", isDirectory: true)
            guard Self.verifyAppCodeSignature(appURL: newAppURL) else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "安全拦截：Apple Developer ID 代码签名校验未通过。"
                return
            }

            statusMessage = "校验通过，正在无缝热替换并重启输入法…"
            let backupURL = targetBundleURL.deletingLastPathComponent()
                .appendingPathComponent("Chime.app.old-\(ProcessInfo.processInfo.processIdentifier)")
            try? FileManager.default.removeItem(at: backupURL)
            if FileManager.default.fileExists(atPath: targetBundleURL.path) {
                try FileManager.default.moveItem(at: targetBundleURL, to: backupURL)
            }
            do {
                try FileManager.default.moveItem(at: newAppURL, to: targetBundleURL)
            } catch {
                // Roll back atomically if moving into target failed
                try? FileManager.default.moveItem(at: backupURL, to: targetBundleURL)
                throw error
            }
            try? FileManager.default.removeItem(at: backupURL)
            try? FileManager.default.removeItem(at: workDir)

            let lsregister = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
            _ = Self.runProcess(lsregister, ["-f", "-R", "-trusted", targetBundleURL.path])

            // Spawn a detached relauncher after 0.35s and exit cleanly so IMKServer reconnects
            // WITHOUT passing `--activate-on-install` (prevents macOS 15+ redundant enable sheets!).
            let relaunch = Process()
            relaunch.executableURL = URL(fileURLWithPath: "/bin/sh")
            let escaped = targetBundleURL.path.replacingOccurrences(of: "'", with: "'\\''")
            relaunch.arguments = ["-c", "sleep 0.35 && /usr/bin/open -n -g -a '\(escaped)'"]
            try? relaunch.run()
            exit(0)
        } catch {
            try? FileManager.default.removeItem(at: workDir)
            statusMessage = "更新失败：\(error.localizedDescription)"
        }
    }

    private func performSystemPkgUpdate(manifest: ChimeReleaseManifest) async {
        let ver = manifest.version
        guard let url = URL(string: "https://github.com/tyrchen/chime/releases/download/v\(ver)/Chime-v\(ver)-macos-arm64.pkg") else {
            statusMessage = "无效的安装包地址。"
            return
        }
        statusMessage = "正在下载 Chime v\(ver) 系统安装包 (.pkg)…"
        let session = Self.makeZeroTelemetrySession()
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chime-pkg-\(UUID().uuidString)", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
            let (tmpFile, resp) = try await session.download(from: url)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "下载安装包失败（HTTP 状态异常）。"
                return
            }

            let pkgURL = workDir.appendingPathComponent("Chime-v\(ver)-macos-arm64.pkg")
            try FileManager.default.moveItem(at: tmpFile, to: pkgURL)

            statusMessage = "正在校验 SHA-256 与 Apple 公证安装器签名…"
            let actualHash = try Self.sha256Hex(of: pkgURL)
            guard actualHash == manifest.pkg_sha256.lowercased() else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "安全拦截：安装包 SHA-256 校验不匹配。"
                return
            }

            guard Self.verifyPkgSignature(pkgURL: pkgURL) else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "安全拦截：Developer ID Installer 签名校验未通过。"
                return
            }

            // Open the SHA-256 + Developer ID verified `.pkg` in macOS native Installer.app, which
            // natively supports Touch ID / Apple Watch authentication (unlike legacy AppleScript
            // `osascript ... with administrator privileges` which only accepts typed passwords!).
            statusMessage = "已通过安全校验，正在打开 macOS 安装器（支持 Touch ID 指纹授权）…"
            let opened = NSWorkspace.shared.open(pkgURL)
            if !opened {
                statusMessage = "无法自动打开安装器，请手动双击：\(pkgURL.path)"
            }
        } catch {
            try? FileManager.default.removeItem(at: workDir)
            statusMessage = "下载更新失败：\(error.localizedDescription)"
        }
    }

    /// Downloads, verifies (`SHA-256`), and hot-mounts an official `.imepack` cartridge with zero restart.
    public func installOrUpdateOfficialPack(packId: String, onCompleted: @escaping @MainActor () -> Void) {
        guard activePackDownloadId == nil else { return }
        activePackDownloadId = packId

        Task { @MainActor in
            defer { activePackDownloadId = nil }
            if latestManifest == nil {
                await refreshManifest(userInitiated: true)
            }
            guard let manifest = latestManifest else {
                statusMessage = "无法获取官方词包校验清单，请检查网络连接。"
                return
            }
            if await downloadAndInstallOfficialPack(packId: packId, manifest: manifest) {
                onCompleted()
            }
        }
    }

    @discardableResult
    private func downloadAndInstallOfficialPack(packId: String, manifest: ChimeReleaseManifest) async -> Bool {
        let expectedSha: String
        switch packId {
        case "classics-poetry":
            expectedSha = manifest.classics_sha256.lowercased()
        case "tech-developer":
            expectedSha = manifest.tech_sha256.lowercased()
        default:
            statusMessage = "未知的官方扩展词包：\(packId)"
            return false
        }

        let ver = manifest.version
        guard let url = URL(string: "https://github.com/tyrchen/chime/releases/download/v\(ver)/\(packId).imepack.zip") else {
            return false
        }

        statusMessage = "正在下载扩展包 \(packId) (v\(ver))…"
        let session = Self.makeZeroTelemetrySession()
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chime-pack-\(UUID().uuidString)", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
            let (tmpFile, resp) = try await session.download(from: url)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "下载扩展包 \(packId) 失败。"
                return false
            }

            let zipURL = workDir.appendingPathComponent("\(packId).imepack.zip")
            try FileManager.default.moveItem(at: tmpFile, to: zipURL)

            let actualSha = try Self.sha256Hex(of: zipURL)
            guard actualSha == expectedSha else {
                try? FileManager.default.removeItem(at: workDir)
                statusMessage = "安全拦截：\(packId) 的 SHA-256 校验未通过。"
                return false
            }

            let imported = ChimeEngineBridge.shared.importAsset(url: zipURL)
            try? FileManager.default.removeItem(at: workDir)

            if imported >= 0 {
                CoreMLStatefulReranker.shared.loadInstalledModelIfPresent()
                statusMessage = "已安装并热挂载 \(packId) (v\(ver))，无需重启立即生效！"
                return true
            } else {
                statusMessage = "挂载扩展包 \(packId) 失败。"
                return false
            }
        } catch {
            try? FileManager.default.removeItem(at: workDir)
            statusMessage = "下载扩展包失败：\(error.localizedDescription)"
            return false
        }
    }

    // MARK: - Cryptographic & Semver Helpers

    public nonisolated static func isVersionNewer(_ candidate: String, than baseline: String) -> Bool {
        let parse: (String) -> [Int] = { raw in
            raw.trimmingCharacters(in: CharacterSet(charactersIn: "vV \t\n"))
                .split(separator: "-")
                .first
                .map { $0.split(separator: ".").map { Int($0) ?? 0 } } ?? [0]
        }
        let a = parse(candidate)
        let b = parse(baseline)
        let n = max(a.count, b.count)
        for idx in 0..<n {
            let av = idx < a.count ? a[idx] : 0
            let bv = idx < b.count ? b[idx] : 0
            if av != bv {
                return av > bv
            }
        }
        return false
    }

    private nonisolated static func makeZeroTelemetrySession() -> URLSession {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.httpShouldSetCookies = false
        cfg.httpCookieAcceptPolicy = .never
        cfg.urlCache = nil
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.timeoutIntervalForRequest = 20
        cfg.timeoutIntervalForResource = 180
        return URLSession(configuration: cfg)
    }

    private nonisolated static func sha256Hex(of fileURL: URL) throws -> String {
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private nonisolated static func verifyAppCodeSignature(appURL: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: appURL.path) else { return false }
        let strict = runProcess("/usr/bin/codesign", ["--verify", "--deep", "--strict", appURL.path])
        guard strict.exitCode == 0 else { return false }

        let info = runProcess("/usr/bin/codesign", ["-dv", appURL.path])
        let combined = info.stdout + "\n" + info.stderr
        return combined.contains("TeamIdentifier=\(expectedTeamId)")
            && combined.contains("Identifier=\(expectedBundleId)")
    }

    private nonisolated static func verifyPkgSignature(pkgURL: URL) -> Bool {
        let res = runProcess("/usr/sbin/pkgutil", ["--check-signature", pkgURL.path])
        guard res.exitCode == 0 else { return false }
        return res.stdout.contains("(\(expectedTeamId))")
    }

    private nonisolated static func runProcess(_ executable: String, _ args: [String]) -> (exitCode: Int32, stdout: String, stderr: String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments = args
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        do {
            try proc.run()
            proc.waitUntilExit()
            let outStr = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            let errStr = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return (proc.terminationStatus, outStr, errStr)
        } catch {
            return (-1, "", error.localizedDescription)
        }
    }
}
