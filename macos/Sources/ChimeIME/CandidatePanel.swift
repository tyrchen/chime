import AppKit

/// Interactive borderless, non-activating 120Hz candidate panel supporting:
/// 1. Compact `1x5` horizontal/vertical candidate bar with `[中。/英.]` toggle & `⌄` expand button;
/// 2. Doubao-style `5x5` 2D candidate grid (`↓` key) with phrases + single characters;
/// 3. Copilot-style `⇥` contextual ghost completion pill (`一路 Tab` for poetry, idioms & habits).
@MainActor
public final class CandidatePanel: NSPanel {
    public static let shared = CandidatePanel()

    public var onSelectCandidate: ((Int) -> Void)?
    public var onToggleGrid: (() -> Void)?
    public var onTogglePunctuation: (() -> Void)?

    private let rootStack = NSStackView()

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

    public func update(response: ChimeResponse, caretRect: NSRect) {
        let candidates = response.candidates
        if candidates.isEmpty && response.ghost_text.isEmpty {
            self.orderOut(nil)
            return
        }

        rootStack.arrangedSubviews.forEach { $0.removeFromSuperview() }

        if candidates.isEmpty && !response.ghost_text.isEmpty {
            rootStack.addArrangedSubview(buildGhostCompletionPill(ghostText: response.ghost_text))
        } else if response.grid_expanded {
            buildExpandedGrid(response: response)
        } else {
            buildCompactBar(response: response)
        }

        rootStack.layoutSubtreeIfNeeded()
        let fitting = rootStack.fittingSize
        let origin = clampedOrigin(size: fitting, caretRect: caretRect)
        self.setFrame(NSRect(origin: origin, size: fitting), display: true)
        self.orderFrontRegardless()
    }

    public func hide() {
        self.orderOut(nil)
    }

    // MARK: - 1. Ghost Completion Pill (`一路 Tab`)

    private func buildGhostCompletionPill(ghostText: String) -> NSView {
        let container = ClickableActionView { [weak self] in
            self?.onSelectCandidate?(0)
        }
        let row = NSStackView()
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        row.translatesAutoresizingMaskIntoConstraints = false

        let iconBadge = makeTagBadge(text: "⇥ 联想", accent: true)
        let textLabel = NSTextField(labelWithString: ghostText)
        textLabel.font = NSFont.systemFont(ofSize: 14, weight: .medium)
        textLabel.textColor = NSColor.labelColor

        let hintLabel = NSTextField(labelWithString: "Tab 补全 · Esc 忽略")
        hintLabel.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        hintLabel.textColor = NSColor.tertiaryLabelColor

        row.addArrangedSubview(iconBadge)
        row.addArrangedSubview(textLabel)
        row.addArrangedSubview(hintLabel)

        container.addSubview(row)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 4),
            row.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -4),
            row.topAnchor.constraint(equalTo: container.topAnchor, constant: 2),
            row.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -2)
        ])
        return container
    }

    // MARK: - 2. Compact 1x5 Candidate Bar

    private func buildCompactBar(response: ChimeResponse) {
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
                fixedWidth: nil
            )
            bar.addArrangedSubview(pill)
        }

        let controls = NSStackView()
        controls.orientation = .horizontal
        controls.alignment = .centerY
        controls.spacing = 5

        if response.total_pages > 1 {
            let expandBtn = buildSmallButton(
                title: "⌄",
                tooltip: "按下键 ↓ 展开 5×5 候选面板"
            ) { [weak self] in
                self?.onToggleGrid?()
            }
            controls.addArrangedSubview(expandBtn)
        }

        let punctBtn = buildSmallButton(
            title: response.chinese_punctuation ? "中。" : "英.",
            tooltip: "点击或按 Ctrl+. 切换中英文标点"
        ) { [weak self] in
            self?.onTogglePunctuation?()
        }
        controls.addArrangedSubview(punctBtn)

        bar.addArrangedSubview(controls)
        rootStack.addArrangedSubview(bar)
    }

    // MARK: - 3. Doubao-style 5x5 Expanded Grid

    private func buildExpandedGrid(response: ChimeResponse) {
        // Header row: preedit + page counter + punctuation toggle + collapse button
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

        // 5x5 Candidate Grid (up to 5 rows x 5 columns)
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
                        fixedWidth: 96
                    )
                    rowStack.addArrangedSubview(cell)
                } else {
                    let empty = NSView()
                    empty.translatesAutoresizingMaskIntoConstraints = false
                    NSLayoutConstraint.activate([
                        empty.widthAnchor.constraint(greaterThanOrEqualToConstant: 96),
                        empty.heightAnchor.constraint(equalToConstant: 28)
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
        fixedWidth: CGFloat?
    ) -> NSView {
        let container = ClickableCandidateView(candidateIndex: clickIndex) { [weak self] clickedIdx in
            self?.onSelectCandidate?(clickedIdx)
        }
        container.wantsLayer = true
        container.layer?.cornerRadius = 6.0
        if isSelected {
            container.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.20).cgColor
        } else {
            container.layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.05).cgColor
        }

        let label = NSTextField(
            labelWithAttributedString: formatCandidate(
                displayNumber: displayNumber,
                item: item,
                isSelected: isSelected
            )
        )
        label.lineBreakMode = .byTruncatingTail
        label.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(label)

        var constraints = [
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 7),
            label.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -7),
            label.topAnchor.constraint(equalTo: container.topAnchor, constant: 4),
            label.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -4)
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
        isSelected: Bool
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        if let num = displayNumber {
            let indexAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium),
                .foregroundColor: isSelected ? NSColor.controlAccentColor : NSColor.secondaryLabelColor
            ]
            result.append(NSAttributedString(string: "\(num). ", attributes: indexAttrs))
        }

        var wordAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14, weight: isSelected ? .semibold : .regular),
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
        let btn = ClickableActionView(onClick: action)
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

    private func clampedOrigin(size: NSSize, caretRect: NSRect) -> NSPoint {
        let screen = NSScreen.screens.first(where: { NSMouseInRect(caretRect.origin, $0.frame, false) })
            ?? NSScreen.main
        let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)

        var x = caretRect.minX
        var y = caretRect.minY - size.height - 6.0
        if y < visible.minY {
            y = caretRect.maxY + 6.0
        }
        if x + size.width > visible.maxX {
            x = max(visible.minX + 8.0, visible.maxX - size.width - 8.0)
        }
        return NSPoint(x: x, y: y)
    }
}

@MainActor
private final class ClickableCandidateView: NSView {
    private let candidateIndex: Int
    private let onClick: (Int) -> Void

    init(candidateIndex: Int, onClick: @escaping (Int) -> Void) {
        self.candidateIndex = candidateIndex
        self.onClick = onClick
        super.init(frame: .zero)
        self.translatesAutoresizingMaskIntoConstraints = false
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func mouseDown(with event: NSEvent) {
        onClick(candidateIndex)
    }
}

@MainActor
private final class ClickableActionView: NSView {
    private let onClick: () -> Void

    init(onClick: @escaping () -> Void) {
        self.onClick = onClick
        super.init(frame: .zero)
        self.translatesAutoresizingMaskIntoConstraints = false
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func mouseDown(with event: NSEvent) {
        onClick()
    }
}
