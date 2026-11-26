import AppKit
import QuartzCore

/// Interactive borderless, non-activating 120Hz candidate panel supporting:
/// 1. Compact `1x5` horizontal/vertical candidate bar with `[中/英]` & `[中。/英.]` toggles + `⌄` expand;
/// 2. Doubao-style `5x5` 2D candidate grid (`↓` key) with phrases + single characters;
/// 3. Copilot-style `⇥` contextual ghost completion pill (`一路 Tab` for poetry, idioms & habits).
///
/// Anchoring stability guarantees:
/// - Locks a vertical baseline + above/below orientation for the entire active composition session
///   so per-keystroke `lineHeightRectangle` jitter (1-14pt) NEVER causes vertical jumping or
///   above/below flip-flopping;
/// - Enforces a constant row height across every compact-bar frame regardless of English descenders,
///   typo underlines, `⇥` badges, or `⌄` button presence;
/// - Supports `preservePosition: true` for asynchronous Route A upgrades so background candidate
///   updates never shift the panel even 1px.
@MainActor
public final class CandidatePanel: NSPanel {
    public static let shared = CandidatePanel()

    public var onSelectCandidate: ((Int) -> Void)?
    public var onToggleGrid: (() -> Void)?
    public var onTogglePunctuation: (() -> Void)?
    public var onToggleEnglish: (() -> Void)?

    private let rootStack = NSStackView()

    // Composition-session anchor state (cleared on `hide()`)
    private var sessionAnchorCaret: NSRect?
    private var sessionPlacedAbove: Bool?
    private var lastAnchorScreenFrame: NSRect?

    /// Returns the currently locked composition caret anchor if the panel is visible.
    public var activeAnchorCaret: NSRect? {
        isVisible ? sessionAnchorCaret : nil
    }

    private init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 40),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true
        )
        self.level = .popUpMenu
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let vibrancy = NSVisualEffectView()
        vibrancy.material = .popover
        vibrancy.blendingMode = .behindWindow
        vibrancy.state = .active
        vibrancy.wantsLayer = true
        vibrancy.layer?.cornerRadius = 11.0
        vibrancy.layer?.masksToBounds = true
        vibrancy.layer?.borderWidth = 0.5
        vibrancy.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor

        rootStack.orientation = .vertical
        rootStack.alignment = .leading
        rootStack.spacing = 6
        rootStack.edgeInsets = NSEdgeInsets(top: 7, left: 10, bottom: 7, right: 10)
        rootStack.translatesAutoresizingMaskIntoConstraints = false

        vibrancy.addSubview(rootStack)
        NSLayoutConstraint.activate([
            rootStack.leadingAnchor.constraint(equalTo: vibrancy.leadingAnchor),
            rootStack.trailingAnchor.constraint(equalTo: vibrancy.trailingAnchor),
            rootStack.topAnchor.constraint(equalTo: vibrancy.topAnchor),
            rootStack.bottomAnchor.constraint(equalTo: vibrancy.bottomAnchor)
        ])
        self.contentView = vibrancy
    }

    override public var canBecomeKey: Bool { false }
    override public var canBecomeMain: Bool { false }

    public func update(response: ChimeResponse, caretRect: NSRect, preservePosition: Bool = false) {
        let candidates = response.candidates
        if candidates.isEmpty && response.ghost_text.isEmpty {
            hide()
            return
        }
        // Dismiss any lingering mode-switch HUD pill the moment candidates appear so they never overlap!
        StatusToastPanel.shared.hide()

        let wasVisible = self.isVisible
        let previousFrame = self.frame

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }

        rootStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        let fontSize = CGFloat(max(13, min(20, response.candidate_font_size)))
        let lockedCellHeight = ceil(fontSize * 1.45) + 8.0

        if candidates.isEmpty && !response.ghost_text.isEmpty {
            rootStack.addArrangedSubview(
                buildGhostCompletionPill(ghostText: response.ghost_text, fontSize: fontSize, cellHeight: lockedCellHeight)
            )
        } else if response.grid_expanded {
            buildExpandedGrid(response: response, fontSize: fontSize, cellHeight: lockedCellHeight)
        } else {
            buildCompactBar(response: response, fontSize: fontSize, cellHeight: lockedCellHeight)
        }

        rootStack.layoutSubtreeIfNeeded()
        let rawFitting = rootStack.fittingSize
        let fitting = NSSize(width: ceil(rawFitting.width), height: ceil(rawFitting.height))

        let origin: NSPoint
        if preservePosition && wasVisible, let lockedAnchor = sessionAnchorCaret {
            // Async Route A refresh: keep the exact vertical anchor & top-left horizontal origin
            // (only nudge left if the expanded width would clip the right screen edge).
            let screen = Self.screen(for: lockedAnchor.origin)
            let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let placedAbove = sessionPlacedAbove ?? false
            var x = previousFrame.origin.x
            if x + fitting.width > visible.maxX - 8.0 {
                x = max(visible.minX + 8.0, visible.maxX - fitting.width - 8.0)
            }
            if x < visible.minX + 8.0 {
                x = visible.minX + 8.0
            }
            let y: CGFloat = if placedAbove {
                min(visible.maxY - fitting.height - 6.0, lockedAnchor.maxY + 6.0)
            } else {
                max(visible.minY + 6.0, lockedAnchor.minY - fitting.height - 6.0)
            }
            origin = NSPoint(x: round(x), y: round(y))
        } else {
            let effectiveAnchor = resolveSessionAnchor(incomingCaret: caretRect, wasVisible: wasVisible)
            let (computedOrigin, placedAbove) = Self.anchoredOrigin(
                size: fitting,
                caretRect: effectiveAnchor,
                lockedPlacedAbove: wasVisible ? sessionPlacedAbove : nil
            )
            sessionAnchorCaret = effectiveAnchor
            sessionPlacedAbove = placedAbove
            lastAnchorScreenFrame = Self.screen(for: effectiveAnchor.origin)?.frame
            origin = computedOrigin
        }

        let targetFrame = NSRect(origin: origin, size: fitting)
        if !wasVisible || abs(previousFrame.origin.x - targetFrame.origin.x) > 0.5
            || abs(previousFrame.origin.y - targetFrame.origin.y) > 0.5
            || abs(previousFrame.size.width - targetFrame.size.width) > 0.5
            || abs(previousFrame.size.height - targetFrame.size.height) > 0.5
        {
            self.setFrame(targetFrame, display: true)
        } else {
            self.contentView?.needsDisplay = true
        }
        if !wasVisible {
            self.orderFrontRegardless()
        }
    }

    public func hide() {
        sessionAnchorCaret = nil
        sessionPlacedAbove = nil
        lastAnchorScreenFrame = nil
        if self.isVisible {
            self.orderOut(nil)
        }
    }

    /// Stabilizes the vertical caret anchor across consecutive keystrokes within the same
    /// composition session so 1-14pt client `lineHeightRectangle` jitter never makes the panel hop.
    private func resolveSessionAnchor(incomingCaret: NSRect, wasVisible: Bool) -> NSRect {
        guard wasVisible, let locked = sessionAnchorCaret else {
            return incomingCaret
        }
        let incomingScreenFrame = Self.screen(for: incomingCaret.origin)?.frame
        if let prevScreen = lastAnchorScreenFrame, let newScreen = incomingScreenFrame, prevScreen != newScreen {
            // User moved across monitors: adopt the new monitor's caret.
            sessionPlacedAbove = nil
            return incomingCaret
        }
        let dy = abs(incomingCaret.minY - locked.minY)
        let dx = abs(incomingCaret.minX - locked.minX)
        // True line wrap or focus jump (> 16pt vertically or > 260pt horizontally): re-anchor smoothly.
        if dy > 16.0 || dx > 260.0 {
            return incomingCaret
        }
        // Same composition line: keep locked vertical baseline (`minY` & `height`) rock-steady and
        // keep horizontal anchor left-pinned to the start of the active preedit.
        return locked
    }

    // MARK: - 1. Ghost Completion Pill (`一路 Tab`)

    private func buildGhostCompletionPill(ghostText: String, fontSize: CGFloat, cellHeight: CGFloat) -> NSView {
        let container = ClickableActionView(hoverEnabled: true) { [weak self] in
            self?.onSelectCandidate?(0)
        }
        let row = NSStackView()
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        row.translatesAutoresizingMaskIntoConstraints = false

        let iconBadge = makeTagBadge(text: "⇥ 联想", accent: true)
        let textLabel = NSTextField(labelWithString: ghostText)
        textLabel.font = NSFont.systemFont(ofSize: fontSize, weight: .medium)
        textLabel.textColor = NSColor.labelColor

        let hintLabel = NSTextField(labelWithString: "Tab 补全 · Esc 忽略")
        hintLabel.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        hintLabel.textColor = NSColor.tertiaryLabelColor

        row.addArrangedSubview(iconBadge)
        row.addArrangedSubview(textLabel)
        row.addArrangedSubview(hintLabel)

        container.addSubview(row)
        NSLayoutConstraint.activate([
            container.heightAnchor.constraint(equalToConstant: cellHeight),
            row.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 4),
            row.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -4),
            row.centerYAnchor.constraint(equalTo: container.centerYAnchor)
        ])
        return container
    }

    // MARK: - 2. Compact 1x5 Candidate Bar

    private func buildCompactBar(response: ChimeResponse, fontSize: CGFloat, cellHeight: CGFloat) {
        let isVertical = response.candidate_layout == "vertical"
        let bar = NSStackView()
        bar.orientation = isVertical ? .vertical : .horizontal
        bar.alignment = isVertical ? .leading : .centerY
        bar.spacing = 6

        for (idx, item) in response.candidates.enumerated() {
            let isSelected = idx == response.highlighted_index
            let pill = buildCandidateCell(
                clickIndex: idx,
                displayNumber: idx + 1,
                item: item,
                isSelected: isSelected,
                fontSize: fontSize,
                fixedWidth: nil,
                fixedHeight: cellHeight
            )
            bar.addArrangedSubview(pill)
        }

        if response.total_pages > 1 {
            let controls = NSStackView()
            controls.orientation = .horizontal
            controls.alignment = .centerY
            controls.spacing = 5
            controls.translatesAutoresizingMaskIntoConstraints = false
            controls.heightAnchor.constraint(equalToConstant: cellHeight).isActive = true

            let expandBtn = buildSmallButton(
                title: "⌄",
                tooltip: "按下键 ↓ 展开 5×5 候选面板"
            ) { [weak self] in
                self?.onToggleGrid?()
            }
            controls.addArrangedSubview(expandBtn)
            bar.addArrangedSubview(controls)
        }
        rootStack.addArrangedSubview(bar)
    }

    // MARK: - 3. Doubao-style 5x5 Expanded Grid

    private func buildExpandedGrid(response: ChimeResponse, fontSize: CGFloat, cellHeight: CGFloat) {
        let header = NSStackView()
        header.orientation = .horizontal
        header.alignment = .centerY
        header.spacing = 8

        let preeditLabel = NSTextField(labelWithString: response.preedit_text)
        preeditLabel.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .medium)
        preeditLabel.textColor = NSColor.secondaryLabelColor
        header.addArrangedSubview(preeditLabel)

        let spacer = NSView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        header.addArrangedSubview(spacer)

        if response.total_pages > 1 {
            let pageText = "\(response.page_index + 1)/\(response.total_pages)  (-/= 翻页)"
            let pageBadge = NSTextField(labelWithString: pageText)
            pageBadge.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular)
            pageBadge.textColor = NSColor.tertiaryLabelColor
            header.addArrangedSubview(pageBadge)
        }

        let modeBtn = buildSmallButton(
            title: response.inline_english_mode ? "英" : "中",
            tooltip: "点击或按 Shift 切换中英文输入模式"
        ) { [weak self] in
            self?.onToggleEnglish?()
        }
        header.addArrangedSubview(modeBtn)

        let punctBtn = buildSmallButton(
            title: response.chinese_punctuation ? "中。" : "英.",
            tooltip: "点击或按 Ctrl+. 切换中英文标点"
        ) { [weak self] in
            self?.onTogglePunctuation?()
        }
        header.addArrangedSubview(punctBtn)

        let collapseBtn = buildSmallButton(
            title: "⌃ 收起",
            tooltip: "按上键 ↑ 或 Esc 收起面板"
        ) { [weak self] in
            self?.onToggleGrid?()
        }
        header.addArrangedSubview(collapseBtn)

        rootStack.addArrangedSubview(header)

        let candidates = response.candidates
        let activeRow = response.highlighted_index / 5
        let totalRows = (candidates.count + 4) / 5

        for r in 0..<min(5, totalRows) {
            let rowStack = NSStackView()
            rowStack.orientation = .horizontal
            rowStack.alignment = .centerY
            rowStack.distribution = .fillEqually
            rowStack.spacing = 6

            for c in 0..<5 {
                let idx = r * 5 + c
                if idx < candidates.count {
                    let isSelected = idx == response.highlighted_index
                    let numberBadge: Int? = (r == activeRow) ? (c + 1) : nil
                    let cell = buildCandidateCell(
                        clickIndex: idx,
                        displayNumber: numberBadge,
                        item: candidates[idx],
                        isSelected: isSelected,
                        fontSize: fontSize,
                        fixedWidth: 96,
                        fixedHeight: cellHeight
                    )
                    rowStack.addArrangedSubview(cell)
                } else {
                    let empty = NSView()
                    empty.translatesAutoresizingMaskIntoConstraints = false
                    NSLayoutConstraint.activate([
                        empty.widthAnchor.constraint(greaterThanOrEqualToConstant: 96),
                        empty.heightAnchor.constraint(equalToConstant: cellHeight)
                    ])
                    rowStack.addArrangedSubview(empty)
                }
            }
            rootStack.addArrangedSubview(rowStack)
        }
    }

    // MARK: - Helpers

    private func buildCandidateCell(
        clickIndex: Int,
        displayNumber: Int?,
        item: ChimeCandidate,
        isSelected: Bool,
        fontSize: CGFloat,
        fixedWidth: CGFloat?,
        fixedHeight: CGFloat
    ) -> NSView {
        let baseBg = isSelected
            ? NSColor.controlAccentColor.withAlphaComponent(0.20)
            : NSColor.quaternaryLabelColor.withAlphaComponent(0.05)
        let hoverBg = isSelected
            ? NSColor.controlAccentColor.withAlphaComponent(0.28)
            : NSColor.quaternaryLabelColor.withAlphaComponent(0.16)
        let container = ClickableCandidateView(
            candidateIndex: clickIndex,
            normalColor: baseBg,
            hoverColor: hoverBg
        ) { [weak self] clickedIdx in
            self?.onSelectCandidate?(clickedIdx)
        }
        container.wantsLayer = true
        container.layer?.cornerRadius = 6.0
        container.layer?.backgroundColor = baseBg.cgColor

        let label = NSTextField(
            labelWithAttributedString: formatCandidate(
                displayNumber: displayNumber,
                item: item,
                isSelected: isSelected,
                fontSize: fontSize
            )
        )
        label.lineBreakMode = .byTruncatingTail
        label.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(label)

        var constraints = [
            container.heightAnchor.constraint(equalToConstant: fixedHeight),
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 7),
            label.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -7),
            label.centerYAnchor.constraint(equalTo: container.centerYAnchor)
        ]
        if let fixedWidth {
            constraints.append(container.widthAnchor.constraint(greaterThanOrEqualToConstant: fixedWidth))
        }
        NSLayoutConstraint.activate(constraints)
        return container
    }

    private func formatCandidate(
        displayNumber: Int?,
        item: ChimeCandidate,
        isSelected: Bool,
        fontSize: CGFloat
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let idxSize = max(11, fontSize - 2)
        if let num = displayNumber {
            let indexAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedDigitSystemFont(ofSize: idxSize, weight: .medium),
                .foregroundColor: isSelected ? NSColor.controlAccentColor : NSColor.secondaryLabelColor
            ]
            result.append(NSAttributedString(string: "\(num). ", attributes: indexAttrs))
        }

        var wordAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: fontSize, weight: isSelected ? .semibold : .regular),
            .foregroundColor: NSColor.labelColor
        ]
        if item.kind == "TypoCorrected" {
            wordAttrs[.underlineStyle] = NSUnderlineStyle.patternDot.rawValue | NSUnderlineStyle.single.rawValue
            wordAttrs[.underlineColor] = NSColor.systemOrange
        }
        result.append(NSAttributedString(string: item.text, attributes: wordAttrs))

        if item.kind == "Completion" {
            let tabAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 11, weight: .medium),
                .foregroundColor: NSColor.controlAccentColor
            ]
            result.append(NSAttributedString(string: " ⇥", attributes: tabAttrs))
        }
        return result
    }

    private func makeTagBadge(text: String, accent: Bool) -> NSView {
        let box = NSView()
        box.wantsLayer = true
        box.layer?.cornerRadius = 5.0
        box.layer?.backgroundColor = (accent ? NSColor.controlAccentColor : NSColor.secondaryLabelColor)
            .withAlphaComponent(0.16).cgColor
        box.translatesAutoresizingMaskIntoConstraints = false

        let lbl = NSTextField(labelWithString: text)
        lbl.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        lbl.textColor = accent ? NSColor.controlAccentColor : NSColor.secondaryLabelColor
        lbl.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(lbl)
        NSLayoutConstraint.activate([
            lbl.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 5),
            lbl.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -5),
            lbl.topAnchor.constraint(equalTo: box.topAnchor, constant: 2),
            lbl.bottomAnchor.constraint(equalTo: box.bottomAnchor, constant: -2)
        ])
        return box
    }

    private func buildSmallButton(
        title: String,
        tooltip: String,
        action: @escaping () -> Void
    ) -> NSView {
        let btn = ClickableActionView(hoverEnabled: true, onClick: action)
        btn.toolTip = tooltip
        btn.wantsLayer = true
        btn.layer?.cornerRadius = 5.0
        btn.layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.18).cgColor

        let lbl = NSTextField(labelWithString: title)
        lbl.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        lbl.textColor = NSColor.secondaryLabelColor
        lbl.translatesAutoresizingMaskIntoConstraints = false
        btn.addSubview(lbl)
        NSLayoutConstraint.activate([
            lbl.leadingAnchor.constraint(equalTo: btn.leadingAnchor, constant: 6),
            lbl.trailingAnchor.constraint(equalTo: btn.trailingAnchor, constant: -6),
            lbl.topAnchor.constraint(equalTo: btn.topAnchor, constant: 3),
            lbl.bottomAnchor.constraint(equalTo: btn.bottomAnchor, constant: -3)
        ])
        return btn
    }

    fileprivate static func screen(for point: NSPoint) -> NSScreen? {
        NSScreen.screens.first(where: { NSMouseInRect(point, $0.frame, false) }) ?? NSScreen.main
    }

    fileprivate static func anchoredOrigin(
        size: NSSize,
        caretRect: NSRect,
        lockedPlacedAbove: Bool?
    ) -> (NSPoint, Bool) {
        let screen = screen(for: caretRect.origin)
        let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)

        var x = caretRect.minX
        if x + size.width > visible.maxX - 8.0 {
            x = visible.maxX - size.width - 8.0
        }
        if x < visible.minX + 8.0 {
            x = visible.minX + 8.0
        }

        let belowY = caretRect.minY - size.height - 6.0
        let aboveY = caretRect.maxY + 6.0

        let placedAbove: Bool
        if let locked = lockedPlacedAbove {
            // Respect locked orientation unless the panel would physically clip off-screen
            // (e.g. expanding from compact bar to 5x5 grid near the bottom edge).
            if locked {
                placedAbove = (aboveY + size.height <= visible.maxY - 4.0) || (belowY < visible.minY + 4.0)
            } else {
                placedAbove = belowY < visible.minY + 4.0 && (aboveY + size.height <= visible.maxY - 4.0)
            }
        } else {
            placedAbove = belowY < visible.minY + 4.0
        }

        var y = placedAbove ? aboveY : belowY
        if y + size.height > visible.maxY - 4.0 {
            y = visible.maxY - size.height - 4.0
        }
        if y < visible.minY + 4.0 {
            y = visible.minY + 4.0
        }
        return (NSPoint(x: round(x), y: round(y)), placedAbove)
    }

    fileprivate static func clampedOrigin(size: NSSize, caretRect: NSRect) -> NSPoint {
        anchoredOrigin(size: size, caretRect: caretRect, lockedPlacedAbove: nil).0
    }
}

/// Non-activating Sogou/Doubao-style status HUD pill (`中` / `En` / `中。` / `英.`) shown briefly
/// near the caret when switching mode via `Shift` or `Ctrl+.`.
///
/// Guarantees single-instance, zero-jump presentation: if a toast is already visible and a rapid
/// follow-up toggle arrives, it updates in place without jumping to a second screen location.
@MainActor
public final class StatusToastPanel: NSPanel {
    public static let shared = StatusToastPanel()

    private let label = NSTextField(labelWithString: "中")
    private var dismissTask: Task<Void, Never>?

    private init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 46, height: 32),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true
        )
        self.level = .popUpMenu
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.ignoresMouseEvents = true
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let vibrancy = NSVisualEffectView()
        vibrancy.material = .hudWindow
        vibrancy.blendingMode = .behindWindow
        vibrancy.state = .active
        vibrancy.wantsLayer = true
        vibrancy.layer?.cornerRadius = 8.0
        vibrancy.layer?.masksToBounds = true
        vibrancy.layer?.borderWidth = 0.5
        vibrancy.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor

        label.font = NSFont.systemFont(ofSize: 14, weight: .semibold)
        label.textColor = NSColor.labelColor
        label.alignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        vibrancy.addSubview(label)

        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: vibrancy.leadingAnchor, constant: 10),
            label.trailingAnchor.constraint(equalTo: vibrancy.trailingAnchor, constant: -10),
            label.topAnchor.constraint(equalTo: vibrancy.topAnchor, constant: 6),
            label.bottomAnchor.constraint(equalTo: vibrancy.bottomAnchor, constant: -6)
        ])
        self.contentView = vibrancy
    }

    override public var canBecomeKey: Bool { false }
    override public var canBecomeMain: Bool { false }

    public func show(badge: String, caretRect: NSRect) {
        dismissTask?.cancel()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        label.stringValue = badge
        self.contentView?.layoutSubtreeIfNeeded()
        let rawFitting = self.contentView?.fittingSize ?? NSSize(width: 44, height: 30)
        let fitting = NSSize(width: ceil(rawFitting.width), height: ceil(rawFitting.height))
        let origin = CandidatePanel.clampedOrigin(size: fitting, caretRect: caretRect)
        self.alphaValue = 1.0
        self.setFrame(NSRect(origin: origin, size: fitting), display: true)
        self.orderFrontRegardless()
        CATransaction.commit()

        dismissTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(600))
            guard !Task.isCancelled, let self else { return }
            NSAnimationContext.runAnimationGroup({ ctx in
                ctx.duration = 0.14
                self.animator().alphaValue = 0.0
            }, completionHandler: { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self, self.alphaValue < 0.05 else { return }
                    self.orderOut(nil)
                }
            })
        }
    }

    public func hide() {
        dismissTask?.cancel()
        if self.isVisible {
            self.orderOut(nil)
        }
    }
}

@MainActor
private final class ClickableCandidateView: NSView {
    private let candidateIndex: Int
    private let normalColor: NSColor
    private let hoverColor: NSColor
    private let onClick: (Int) -> Void

    init(
        candidateIndex: Int,
        normalColor: NSColor,
        hoverColor: NSColor,
        onClick: @escaping (Int) -> Void
    ) {
        self.candidateIndex = candidateIndex
        self.normalColor = normalColor
        self.hoverColor = hoverColor
        self.onClick = onClick
        super.init(frame: .zero)
        self.translatesAutoresizingMaskIntoConstraints = false
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas {
            removeTrackingArea(area)
        }
        addTrackingArea(NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil
        ))
    }

    override func mouseEntered(with event: NSEvent) {
        layer?.backgroundColor = hoverColor.cgColor
    }

    override func mouseExited(with event: NSEvent) {
        layer?.backgroundColor = normalColor.cgColor
    }

    override func mouseDown(with event: NSEvent) {
        onClick(candidateIndex)
    }
}

@MainActor
private final class ClickableActionView: NSView {
    private let hoverEnabled: Bool
    private let onClick: () -> Void

    init(hoverEnabled: Bool = false, onClick: @escaping () -> Void) {
        self.hoverEnabled = hoverEnabled
        self.onClick = onClick
        super.init(frame: .zero)
        self.translatesAutoresizingMaskIntoConstraints = false
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        guard hoverEnabled else { return }
        for area in trackingAreas {
            removeTrackingArea(area)
        }
        addTrackingArea(NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil
        ))
    }

    override func mouseEntered(with event: NSEvent) {
        guard hoverEnabled else { return }
        layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.32).cgColor
    }

    override func mouseExited(with event: NSEvent) {
        guard hoverEnabled else { return }
        layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.18).cgColor
    }

    override func mouseDown(with event: NSEvent) {
        onClick()
    }
}
