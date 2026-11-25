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

// 1. Vector monochrome template `menu_icon.pdf` (16x16pt pure CGPath even-odd cutout, zero embedded fonts)
var mediaBox = CGRect(x: 0, y: 0, width: 16, height: 16)
guard let pdfConsumer = CGDataConsumer(url: pdfPath as CFURL),
      let pdfCtx = CGContext(consumer: pdfConsumer, mediaBox: &mediaBox, nil)
else {
    fputs("Failed to create PDF context\n", stderr)
    exit(1)
}

pdfCtx.beginPDFPage(nil)
pdfCtx.setFillColor(CGColor(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 1.0))

// Outer solid squircle badge (14x14 centered in 16x16) + inner cutout "C" + chime bar using even-odd rule!
let compound = CGMutablePath()
compound.addRoundedRect(in: CGRect(x: 1.0, y: 1.0, width: 14.0, height: 14.0), cornerWidth: 3.6, cornerHeight: 3.6)

// Cutout geometric "C" arc (center (7.4, 8.0), outer r=4.1, inner r=2.5, opening on right)
let cx: CGFloat = 7.3
let cy: CGFloat = 8.0
let rOut: CGFloat = 4.1
let rIn: CGFloat = 2.45
let startAngle: CGFloat = 0.62 // ~35.5 deg
let endAngle: CGFloat = .pi * 2.0 - 0.62

let cPath = CGMutablePath()
cPath.addArc(center: CGPoint(x: cx, y: cy), radius: rOut, startAngle: startAngle, endAngle: endAngle, clockwise: false)
cPath.addArc(center: CGPoint(x: cx, y: cy), radius: rIn, startAngle: endAngle, endAngle: startAngle, clockwise: true)
cPath.closeSubpath()
compound.addPath(cPath)

// Cutout vertical chime resonance bar on the right
compound.addRoundedRect(in: CGRect(x: 11.4, y: 5.4, width: 1.55, height: 5.2), cornerWidth: 0.75, cornerHeight: 0.75)

pdfCtx.addPath(compound)
pdfCtx.fillPath(using: .evenOdd)
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

    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = .center
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
