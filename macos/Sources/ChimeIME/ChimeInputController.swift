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
        let bundleId = (sender as? IMKTextInput)?.bundleIdentifier() ?? ""
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.activate(controllerKey: key, appBundleId: bundleId)
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
        let menu = NSMenu(title: "Chime")
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
        let type = event.type
        let keyCode = event.keyCode
        let shiftDown = event.modifierFlags.contains(.shift)
        let hasCmdOrCtrl = event.modifierFlags.contains(.command)
            || event.modifierFlags.contains(.control)
            || event.modifierFlags.contains(.option)
        let timestamp = event.timestamp
        let ascii: UInt8 = {
            guard let chars = event.characters, let scalar = chars.unicodeScalars.first, scalar.isASCII else {
                return 0
            }
            return UInt8(scalar.value)
        }()
        let secure = IsSecureEventInputEnabled()

        let response: ChimeResponse? = MainActor.assumeIsolated {
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

        if resp.preedit_text.isEmpty {
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
            )
            MainActor.assumeIsolated {
                CandidatePanel.shared.hide()
            }
            return
        }

        let nsPreedit = resp.preedit_text as NSString
        client.setMarkedText(
            resp.preedit_text,
            selectionRange: NSRange(location: nsPreedit.length, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: NSNotFound)
        )

        var lineRect = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &lineRect)
        let validRect = (lineRect.origin.x != 0 || lineRect.origin.y != 0) ? lineRect : nil

        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.updateCandidatePanel(
                controllerKey: controllerKey,
                response: resp,
                observedCaretRect: validRect
            )
        }
    }
}
