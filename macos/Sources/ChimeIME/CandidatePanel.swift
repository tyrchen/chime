import AppKit

/// Interactive borderless, non-activating 120Hz candidate panel supporting horizontal/vertical layout,
/// highlight pill navigation, page badge, mouse click selection, and typo-correction dotted underline.
@MainActor
public final class CandidatePanel: NSPanel {
    public static let shared = CandidatePanel()

    public var onSelectCandidate: ((Int) -> Void)?

    private let stackView = NSStackView()

    private init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 360, height: 38),
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
        vibrancy.layer?.cornerRadius = 10.0
        vibrancy.layer?.masksToBounds = true

        stackView.orientation = .horizontal
        stackView.alignment = .centerY
        stackView.spacing = 6
        stackView.edgeInsets = NSEdgeInsets(top: 6, left: 10, bottom: 6, right: 10)
        stackView.translatesAutoresizingMaskIntoConstraints = false

        vibrancy.addSubview(stackView)
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: vibrancy.leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: vibrancy.trailingAnchor),
            stackView.topAnchor.constraint(equalTo: vibrancy.topAnchor),
            stackView.bottomAnchor.constraint(equalTo: vibrancy.bottomAnchor)
        ])
        self.contentView = vibrancy
    }

    override public var canBecomeKey: Bool { false }
    override public var canBecomeMain: Bool { false }

    public func update(response: ChimeResponse, caretRect: NSRect) {
        let candidates = response.candidates
        guard !candidates.isEmpty else {
            self.orderOut(nil)
            return
        }

        let isVertical = response.candidate_layout == "vertical"
        stackView.orientation = isVertical ? .vertical : .horizontal
        stackView.alignment = isVertical ? .leading : .centerY

        stackView.arrangedSubviews.forEach { $0.removeFromSuperview() }

        for (idx, item) in candidates.enumerated() {
            let isSelected = idx == response.highlighted_index
            let pill = buildCandidatePill(index: idx, item: item, isSelected: isSelected)
            stackView.addArrangedSubview(pill)
        }

        if response.total_pages > 1 {
            let pageText = "\(response.page_index + 1)/\(response.total_pages)"
            let badge = NSTextField(labelWithString: pageText)
            badge.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular)
            badge.textColor = NSColor.tertiaryLabelColor
            stackView.addArrangedSubview(badge)
        }

        stackView.layoutSubtreeIfNeeded()
        let fitting = stackView.fittingSize
        let origin = clampedOrigin(size: fitting, caretRect: caretRect)
        self.setFrame(NSRect(origin: origin, size: fitting), display: true)
        self.orderFrontRegardless()
    }

    public func hide() {
        self.orderOut(nil)
    }

    private func buildCandidatePill(index: Int, item: ChimeCandidate, isSelected: Bool) -> NSView {
        let container = ClickableCandidateView(candidateIndex: index) { [weak self] clickedIdx in
            self?.onSelectCandidate?(clickedIdx)
        }
        container.wantsLayer = true
        container.layer?.cornerRadius = 6.0
        if isSelected {
            container.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.18).cgColor
        }

        let label = NSTextField(labelWithAttributedString: formatCandidate(index: index + 1, item: item, isSelected: isSelected))
        label.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 6),
            label.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -6),
            label.topAnchor.constraint(equalTo: container.topAnchor, constant: 3),
            label.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -3)
        ])
        return container
    }

    private func formatCandidate(index: Int, item: ChimeCandidate, isSelected: Bool) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let indexAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: isSelected ? NSColor.controlAccentColor : NSColor.secondaryLabelColor
        ]
        result.append(NSAttributedString(string: "\(index). ", attributes: indexAttrs))

        var wordAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14, weight: isSelected ? .semibold : .regular),
            .foregroundColor: NSColor.labelColor
        ]
        if item.kind == "TypoCorrected" {
            wordAttrs[.underlineStyle] = NSUnderlineStyle.patternDot.rawValue | NSUnderlineStyle.single.rawValue
            wordAttrs[.underlineColor] = NSColor.systemOrange
        }
        result.append(NSAttributedString(string: item.text, attributes: wordAttrs))
        return result
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
