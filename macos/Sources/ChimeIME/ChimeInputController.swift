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
        let key = ObjectIdentifier(self)
        let (cfg, isEng): (ChimeConfigDTO, Bool) = MainActor.assumeIsolated {
            (
                ChimeEngineBridge.shared.getConfig() ?? .default,
                ChimeEngineBridge.shared.isInlineEnglish(controllerKey: key)
            )
        }

        let menu = NSMenu(title: "Chime")

        let modeTitle = isEng ? "切换至中文输入模式 (中)   ⇧" : "切换至英文输入模式 (英)   ⇧"
        let modeItem = NSMenuItem(
            title: modeTitle,
            action: #selector(toggleEnglishMode(_:)),
            keyEquivalent: ""
        )
        modeItem.target = self
        menu.addItem(modeItem)

        let punctTitle = cfg.chinese_punctuation
            ? "切换至英文半角标点 (.,)   ⌃."
            : "切换至中文全角标点 (。，)   ⌃."
        let punctItem = NSMenuItem(
            title: punctTitle,
            action: #selector(togglePunctuationMode(_:)),
            keyEquivalent: ""
        )
        punctItem.target = self
        menu.addItem(punctItem)

        let panguItem = NSMenuItem(
            title: "中英文与数字前后自动插入空格",
            action: #selector(togglePanguSpacingMode(_:)),
            keyEquivalent: ""
        )
        panguItem.state = cfg.pangu_spacing ? .on : .off
        panguItem.target = self
        menu.addItem(panguItem)

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

    @objc public func toggleEnglishMode(_ sender: Any?) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            let resp = ChimeEngineBridge.shared.toggleEnglish(controllerKey: key)
            ChimeEngineBridge.shared.applyResponseToStoredClient(controllerKey: key, resp: resp)
            if resp.show_status_toast {
                let caret = ChimeEngineBridge.shared.caretRect(for: key)
                StatusToastPanel.shared.show(badge: resp.inline_english_mode ? "英" : "中", caretRect: caret)
            }
        }
    }

    @objc public func togglePunctuationMode(_ sender: Any?) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            let resp = ChimeEngineBridge.shared.togglePunctuation(controllerKey: key)
            ChimeEngineBridge.shared.applyResponseToStoredClient(controllerKey: key, resp: resp)
            if resp.show_status_toast {
                let caret = ChimeEngineBridge.shared.caretRect(for: key)
                StatusToastPanel.shared.show(badge: resp.chinese_punctuation ? "中。" : "英.", caretRect: caret)
            }
        }
    }

    @objc public func togglePanguSpacingMode(_ sender: Any?) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            let resp = ChimeEngineBridge.shared.togglePanguSpacing(controllerKey: key)
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

        let observedLeftContext: String? = {
            if secure || type != .keyDown {
                return nil
            }
            let sel = client.selectedRange()
            guard sel.location != NSNotFound, sel.location > 0 else {
                return nil
            }
            let takeLen = min(sel.location, 96)
            let range = NSRange(location: sel.location - takeLen, length: takeLen)
            if let attr = client.attributedSubstring(from: range), !attr.string.isEmpty {
                return attr.string
            }
            return nil
        }()

        var isModeToggleToast = false
        var isPunctToggleToast = false

        let response: ChimeResponse? = MainActor.assumeIsolated {
            ChimeEngineBridge.shared.registerClient(controllerKey: key, client: unsafeClientObj)
            if let observedLeftContext {
                ChimeEngineBridge.shared.syncSurroundingContext(
                    controllerKey: key,
                    leftContext: observedLeftContext
                )
            }

            if type == .flagsChanged {
                let toggled = ChimeEngineBridge.shared.pollModifierToggle(
                    controllerKey: key,
                    keyCode: keyCode,
                    flags: flags,
                    timestamp: timestamp
                )
                if toggled != nil {
                    isModeToggleToast = true
                }
                return toggled
            }
            guard type == .keyDown else {
                return nil
            }
            if isCtrlPeriod {
                isPunctToggleToast = true
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
        let validRect = applyClientOutput(controllerKey: key, resp: resp, client: client)

        if resp.show_status_toast && (isModeToggleToast || isPunctToggleToast) {
            MainActor.assumeIsolated {
                let rect = validRect ?? ChimeEngineBridge.shared.caretRect(for: key)
                if isModeToggleToast {
                    StatusToastPanel.shared.show(
                        badge: resp.inline_english_mode ? "英" : "中",
                        caretRect: rect
                    )
                } else if isPunctToggleToast {
                    StatusToastPanel.shared.show(
                        badge: resp.chinese_punctuation ? "中。" : "英.",
                        caretRect: rect
                    )
                }
            }
        }
        return resp.handled
    }

    override public func commitComposition(_ sender: Any!) {
        let key = ObjectIdentifier(self)
        MainActor.assumeIsolated {
            ChimeEngineBridge.shared.commitComposition(controllerKey: key)
        }
    }

    @discardableResult
    private func applyClientOutput(
        controllerKey: ObjectIdentifier,
        resp: ChimeResponse,
        client: IMKTextInput
    ) -> NSRect? {
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
                ChimeEngineBridge.shared.recordObservedCaretRect(controllerKey: controllerKey, rect: validRect)
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
            return validRect
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
        return validRect
    }
}
