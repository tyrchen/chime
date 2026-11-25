import AppKit
import CoreGraphics
import Foundation

// Generates vector template `menu_icon.pdf` (18x18pt) and full-res `AppIcon.iconset` PNGs.
let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("Usage: swift generate_icons.swift <menu_icon.pdf> <AppIcon.iconset_dir>\n", stderr)
    exit(1)
}

let pdfPath = URL(fileURLWithPath: args[1])
let iconsetDir = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: iconsetDir, withIntermediateDirectories: true)

// 1. Vector monochrome template `menu_icon.pdf` (18x18pt rounded badge with crisp "磬/C" glyph)
var mediaBox = CGRect(x: 0, y: 0, width: 18, height: 18)
guard let pdfConsumer = CGDataConsumer(url: pdfPath as CFURL),
      let pdfCtx = CGContext(consumer: pdfConsumer, mediaBox: &mediaBox, nil)
else {
    fputs("Failed to create PDF context\n", stderr)
    exit(1)
}

pdfCtx.beginPDFPage(nil)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(cgContext: pdfCtx, flipped: false)

let badgeRect = NSRect(x: 1.5, y: 1.5, width: 15.0, height: 15.0)
let badgePath = NSBezierPath(roundedRect: badgeRect, xRadius: 3.6, yRadius: 3.6)
badgePath.lineWidth = 1.4
NSColor.black.setStroke()
badgePath.stroke()

let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center
let attrs: [NSAttributedString.Key: Any] = [
    .font: NSFont.systemFont(ofSize: 10.5, weight: .bold),
    .foregroundColor: NSColor.black,
    .paragraphStyle: paragraph
]
let glyph = NSString(string: "拼")
let glyphSize = glyph.size(withAttributes: attrs)
let drawRect = NSRect(
    x: (18.0 - glyphSize.width) / 2.0,
    y: (18.0 - glyphSize.height) / 2.0 - 0.3,
    width: glyphSize.width,
    height: glyphSize.height
)
glyph.draw(in: drawRect, withAttributes: attrs)

NSGraphicsContext.restoreGraphicsState()
pdfCtx.endPDFPage()
pdfCtx.closePDF()

// 2. Render macOS Sequoia squircle AppIcon PNGs for `iconutil -c icns`
let sizes: [(String, Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024)
]

for (filename, px) in sizes {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: px,
        pixelsHigh: px,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else { continue }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    let s = CGFloat(px)
    let inset = s * 0.08
    let rect = NSRect(x: inset, y: inset, width: s - inset * 2, height: s - inset * 2)
    let bgPath = NSBezierPath(roundedRect: rect, xRadius: s * 0.22, yRadius: s * 0.22)

    let gradient = NSGradient(
        starting: NSColor(calibratedRed: 0.13, green: 0.16, blue: 0.24, alpha: 1.0),
        ending: NSColor(calibratedRed: 0.20, green: 0.36, blue: 0.62, alpha: 1.0)
    )
    gradient?.draw(in: bgPath, angle: -45.0)

    let textAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: s * 0.48, weight: .bold),
        .foregroundColor: NSColor.white,
        .paragraphStyle: paragraph
    ]
    let label = NSString(string: "拼")
    let tSize = label.size(withAttributes: textAttrs)
    label.draw(
        in: NSRect(
            x: (s - tSize.width) / 2.0,
            y: (s - tSize.height) / 2.0,
            width: tSize.width,
            height: tSize.height
        ),
        withAttributes: textAttrs
    )

    NSGraphicsContext.restoreGraphicsState()
    if let pngData = rep.representation(using: .png, properties: [:]) {
        try pngData.write(to: iconsetDir.appendingPathComponent(filename))
    }
}
