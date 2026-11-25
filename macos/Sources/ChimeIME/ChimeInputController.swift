import AppKit
import Carbon
import InputMethodKit

/// Zero-property `IMKInputController` shell preventing `IMKServer` proxy dictionary memory bloat
/// and avoiding non-Sendable `self` captures across Swift 6 `@MainActor` isolation boundaries.
@objc(ChimeInputController)
public final class ChimeInputController: IMKInputController {
    override public func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        let key = ObjectIdentifier(self)
        let clientObj = sender as AnyObject?
        let bundleId = (sender as? IMKTextInput)?.bundleIdentifier() ?? ""
        nonisolated(unsafe) let unsafeClient = clientObj
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.activate(
                controllerKey: key,
                appBundleId: bundleId,
                client: unsafeClient
            )
        }
    }

    override public func deactivateServer(_ sender: Any!) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.deactivate(controllerKey: key)
        }
        super.deactivateServer(sender)
    }

    override public func recognizedEvents(_ sender: Any!) -> Int {
        let mask: NSEvent.EventTypeMask = [.keyDown, .flagsChanged]
        return Int(mask.rawValue)
    }

    override public func menu() -> NSMenu! {
        let isCnPunct: Bool = MainActor.assumeIsolated {
            ChimeEngineBridge.shared.getConfig()?.chinese_punctuation ?? true
        }

        let menu = NSMenu(title: "Chime")

        let punctTitle = isCnPunct ? "切换至英文半角标点 (.,)   ⌃." : "切换至中文全角标点 (。，)   ⌃."
        let punctItem = NSMenuItem(
            title: punctTitle,
            action: #selector(togglePunctuationMode(_:)),
            keyEquivalent: ""
        )
        punctItem.target = self
        menu.addItem(punctItem)
        menu.addItem(NSMenuItem.separator())

        let prefsItem = NSMenuItem(
            title: "偏好设置…",
            action: #selector(openPreferences(_:)),
            keyEquivalent: ","
        )
        prefsItem.target = self
        menu.addItem(prefsItem)

        let importItem = NSMenuItem(
            title: "导入词库或个人模型包 (.scel / .dict.yaml / .imepack)…",
            action: #selector(importDictionaryOrPack(_:)),
            keyEquivalent: ""
        )
        importItem.target = self
        menu.addItem(importItem)

        let reloadItem = NSMenuItem(
            title: "热重载配置与模型",
            action: #selector(reloadEngineAssets(_:)),
            keyEquivalent: "r"
        )
        reloadItem.target = self
        menu.addItem(reloadItem)

        return menu
    }

    @objc public func togglePunctuationMode(_ sender: Any?) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            let resp = ChimeEngineBridge.shared.togglePunctuation(controllerKey: key)
            ChimeEngineBridge.shared.applyResponseToStoredClient(controllerKey: key, resp: resp)
        }
    }

    @objc public func openPreferences(_ sender: Any?) {
        MainActor.assumeIsolated {
            ChimeSettingsWindowController.shared.showWindow()
        }
    }

    @objc public func importDictionaryOrPack(_ sender: Any?) {
        MainActor.assumeIsolated {
            ChimeSettingsWindowController.shared.presentImportOpenPanel()
        }
    }

    @objc public func reloadEngineAssets(_ sender: Any?) {
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.reloadAssets()
            CoreMLStatefulReranker.shared.loadInstalledModelIfPresent()
        }
    }

    override public func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, let client = sender as? IMKTextInput else {
            return false
        }
        let key = ObjectIdentifier(self)
        nonisolated(unsafe) let unsafeClientObj = sender as AnyObject
        let type = event.type
        let keyCode = event.keyCode
        let flags = event.modifierFlags
        let shiftDown = flags.contains(.shift)
        let ctrlOnly = flags.contains(.control) && !flags.contains(.command) && !flags.contains(.option)
        let hasCmdOrCtrl = flags.contains(.command) || flags.contains(.control) || flags.contains(.option)
        let timestamp = event.timestamp
        let ascii: UInt8 = {
            guard let chars = event.characters, let scalar = chars.unicodeScalars.first, scalar.isASCII else {
                return 0
            }
            return UInt8(scalar.value)
        }()
        let isCtrlPeriod = ctrlOnly && (keyCode == 0x2F || event.charactersIgnoringModifiers == ".")
        let secure = IsSecureEventInputEnabled()

        let response: ChimeResponse? = MainActor.assumeIsolated {
            ChimeEngineBridge.shared.registerClient(controllerKey: key, client: unsafeClientObj)

            if type == .flagsChanged {
                return ChimeEngineBridge.shared.pollLeftShiftTap(
                    controllerKey: key,
                    keyCode: keyCode,
                    shiftDown: shiftDown,
                    timestamp: timestamp
                )
            }
            guard type == .keyDown else {
                return nil
            }
            if isCtrlPeriod {
                return ChimeEngineBridge.shared.togglePunctuation(controllerKey: key)
            }
            return ChimeEngineBridge.shared.processKey(
                controllerKey: key,
                asciiChar: ascii,
                keyCode: keyCode,
                shift: shiftDown,
                hasCommandOrCtrl: hasCmdOrCtrl,
                secureInput: secure
            )
        }

        guard let resp = response else {
            return false
        }
        applyClientOutput(controllerKey: key, resp: resp, client: client)
        return resp.handled
    }

    override public func commitComposition(_ sender: Any!) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.reset(controllerKey: key)
        }
    }

    private func applyClientOutput(
        controllerKey: ObjectIdentifier,
        resp: ChimeResponse,
        client: IMKTextInput
    ) {
        if !resp.committed_text.isEmpty {
            client.insertText(
                resp.committed_text,
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
        }

        var lineRect = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &lineRect)
        let validRect = (lineRect.origin.x != 0 || lineRect.origin.y != 0) ? lineRect : nil

        if resp.preedit_text.isEmpty {
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
            MainActor.assumeIsolated {
                if resp.ghost_text.isEmpty {
                    CandidatePanel.shared.hide()
                } else {
                    ChimeEngineBridge.shared.updateCandidatePanel(
                        controllerKey: controllerKey,
                        response: resp,
                        observedCaretRect: validRect
                    )
                }
            }
            return
        }

        let nsPreedit = resp.preedit_text as NSString
        client.setMarkedText(
            resp.preedit_text,
            selectionRange: NSRange(location: nsPreedit.length, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
        )

        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.updateCandidatePanel(
                controllerKey: controllerKey,
                response: resp,
                observedCaretRect: validRect
            )
        }
    }
}
