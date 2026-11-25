import AppKit

/// Borderless, non-activating 120Hz candidate bar replacing legacy `IMKCandidates`.
@MainActor
public final class CandidatePanel: NSPanel {
    public static let shared = CandidatePanel()

    private let stackView = NSStackView()

    private init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 320, height: 36),
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
        vibrancy.layer?.cornerRadius = 9.0
        vibrancy.layer?.masksToBounds = true

        stackView.orientation = .horizontal
        stackView.alignment = .centerY
        stackView.spacing = 12
        stackView.edgeInsets = NSEdgeInsets(top: 6, left: 12, bottom: 6, right: 12)
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

    public func update(candidates: [ChimeCandidate], caretRect: NSRect) {
        guard !candidates.isEmpty else {
            self.orderOut(nil)
            return
        }

        stackView.arrangedSubviews.forEach { $0.removeFromSuperview() }

        for (idx, item) in candidates.prefix(5).enumerated() {
            let label = NSTextField(labelWithAttributedString: formatCandidate(index: idx + 1, item: item))
            stackView.addArrangedSubview(label)
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

    private func formatCandidate(index: Int, item: ChimeCandidate) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let indexAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: index == 1 ? NSColor.controlAccentColor : NSColor.secondaryLabelColor
        ]
        result.append(NSAttributedString(string: "\(index). ", attributes: indexAttrs))

        var wordAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14, weight: index == 1 ? .semibold : .regular),
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
