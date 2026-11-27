import AppKit

/// Animated level meter (Doubao-style centered bars) shown while listening.
@MainActor
final class VoiceLevelView: NSView {
    private static let envelope: [CGFloat] = [0.45, 0.65, 0.85, 1.0, 0.85, 0.65, 0.45]
    private var history: [CGFloat] = Array(repeating: 0, count: 7)
    var listening = true {
        didSet { needsDisplay = true }
    }

    override var intrinsicContentSize: NSSize {
        NSSize(width: 34, height: 22)
    }

    func push(level: Float) {
        history.removeFirst()
        history.append(CGFloat(max(0, min(1, level))))
        needsDisplay = true
    }

    func reset() {
        history = Array(repeating: 0, count: history.count)
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        let count = Self.envelope.count
        let barWidth: CGFloat = 3
        let gap = (bounds.width - barWidth * CGFloat(count)) / CGFloat(count - 1)
        let color: NSColor = listening ? .controlAccentColor : .secondaryLabelColor
        color.setFill()
        for idx in 0..<count {
            // Mix the latest level with history so neighbouring bars move slightly out of phase.
            let recent = history[history.count - 1 - (idx % 3)]
            let amp = listening ? (0.18 + 0.82 * recent * Self.envelope[idx]) : 0.18
            let height = max(barWidth, bounds.height * amp)
            let x = CGFloat(idx) * (barWidth + gap)
            let rect = NSRect(x: x, y: (bounds.height - height) / 2, width: barWidth, height: height)
            NSBezierPath(roundedRect: rect, xRadius: barWidth / 2, yRadius: barWidth / 2).fill()
        }
    }
}

/// Non-activating floating HUD that shows the live transcript near the caret while dictating.
@MainActor
final class VoiceInputPanel: NSPanel {
    static let shared = VoiceInputPanel()

    private static let maxTextWidth: CGFloat = 420
    private static let maxVisibleChars = 140

    private let levelView = VoiceLevelView()
    private let textLabel = NSTextField(wrappingLabelWithString: "")
    private let hintLabel = NSTextField(labelWithString: "")
    private var anchorRect: NSRect?
    private var hideTask: Task<Void, Never>?

    private init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 240, height: 56),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true
        )
        level = .popUpMenu
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        ignoresMouseEvents = true
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let vibrancy = NSVisualEffectView()
        vibrancy.material = .hudWindow
        vibrancy.blendingMode = .behindWindow
        vibrancy.state = .active
        vibrancy.wantsLayer = true
        vibrancy.layer?.cornerRadius = 12
        vibrancy.layer?.masksToBounds = true
        vibrancy.layer?.borderWidth = 0.5
        vibrancy.layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.35).cgColor

        textLabel.font = NSFont.systemFont(ofSize: 15)
        textLabel.textColor = .labelColor
        textLabel.maximumNumberOfLines = 4
        textLabel.preferredMaxLayoutWidth = Self.maxTextWidth
        textLabel.lineBreakMode = .byWordWrapping
        hintLabel.font = NSFont.systemFont(ofSize: 11)
        hintLabel.textColor = .secondaryLabelColor
        hintLabel.lineBreakMode = .byTruncatingTail

        let column = NSStackView(views: [textLabel, hintLabel])
        column.orientation = .vertical
        column.alignment = .leading
        column.spacing = 3
        let row = NSStackView(views: [levelView, column])
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 10
        row.translatesAutoresizingMaskIntoConstraints = false
        vibrancy.addSubview(row)

        NSLayoutConstraint.activate([
            levelView.widthAnchor.constraint(equalToConstant: 34),
            levelView.heightAnchor.constraint(equalToConstant: 22),
            textLabel.widthAnchor.constraint(lessThanOrEqualToConstant: Self.maxTextWidth),
            hintLabel.widthAnchor.constraint(lessThanOrEqualToConstant: Self.maxTextWidth),
            column.widthAnchor.constraint(greaterThanOrEqualToConstant: 170),
            row.leadingAnchor.constraint(equalTo: vibrancy.leadingAnchor, constant: 14),
            row.trailingAnchor.constraint(equalTo: vibrancy.trailingAnchor, constant: -14),
            row.topAnchor.constraint(equalTo: vibrancy.topAnchor, constant: 10),
            row.bottomAnchor.constraint(equalTo: vibrancy.bottomAnchor, constant: -10)
        ])
        contentView = vibrancy
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// Shows the panel for a new recording anchored below `anchor` (the caret).
    func present(anchor: NSRect, hint: String) {
        hideTask?.cancel()
        anchorRect = anchor
        levelView.reset()
        levelView.listening = true
        apply(text: "", hint: hint, isError: false)
    }

    func update(text: String, level: Float, hint: String, isError: Bool, listening: Bool) {
        levelView.listening = listening
        levelView.push(level: level)
        apply(text: text, hint: hint, isError: isError)
    }

    /// Shows a short message, then fades out.
    func flash(message: String, anchor: NSRect?, isError: Bool) {
        hideTask?.cancel()
        if let anchor {
            anchorRect = anchor
        }
        if anchorRect == nil {
            let mouse = NSEvent.mouseLocation
            anchorRect = NSRect(x: mouse.x, y: mouse.y - 18, width: 1, height: 18)
        }
        levelView.listening = false
        levelView.reset()
        apply(text: message, hint: "", isError: isError)
        dismiss(after: .milliseconds(isError ? 3200 : 1600))
    }

    func dismiss(after delay: Duration = .zero) {
        hideTask?.cancel()
        hideTask = Task { @MainActor [weak self] in
            if delay > .zero {
                try? await Task.sleep(for: delay)
            }
            guard let self, !Task.isCancelled else { return }
            NSAnimationContext.runAnimationGroup({ ctx in
                ctx.duration = 0.12
                self.animator().alphaValue = 0
            }, completionHandler: { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self, self.alphaValue < 0.05 else { return }
                    self.orderOut(nil)
                    self.anchorRect = nil
                }
            })
        }
    }

    private func apply(text: String, hint: String, isError: Bool) {
        let shown: String = if text.isEmpty {
            "正在聆听…"
        } else if text.count > Self.maxVisibleChars {
            "…" + String(text.suffix(Self.maxVisibleChars))
        } else {
            text
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        textLabel.stringValue = shown
        textLabel.textColor = isError && hint.isEmpty
            ? .systemRed
            : (text.isEmpty ? .secondaryLabelColor : .labelColor)
        hintLabel.stringValue = hint
        hintLabel.isHidden = hint.isEmpty
        hintLabel.textColor = isError ? .systemRed : .secondaryLabelColor
        contentView?.layoutSubtreeIfNeeded()
        let fitting = contentView?.fittingSize ?? NSSize(width: 240, height: 56)
        let size = NSSize(width: ceil(fitting.width), height: ceil(fitting.height))
        let anchor = anchorRect ?? NSRect(x: 480, y: 360, width: 1, height: 18)
        let origin = CandidatePanel.clampedOrigin(size: size, caretRect: anchor)
        alphaValue = 1
        setFrame(NSRect(origin: origin, size: size), display: true)
        orderFrontRegardless()
        CATransaction.commit()
    }
}