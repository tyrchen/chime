import Foundation

/// How a review can safely act on this utterance. An opaque editor can still show the HUD;
/// never guess at a backwards replacement in a client that won't confirm the original text.
enum VoiceReviewSite: Equatable {
    case marked(NSRange)
    case committed(NSRange)
    case copyOnly

    static func probe(text: String, start: Int?, marked: NSRange, selected: NSRange,
                      read: (NSRange) -> String?) -> Self {
        let length = text.utf16.count
        guard length > 0 else { return .copyOnly }
        if marked.location != NSNotFound, marked.length == length,
           start == nil || start == marked.location {
            if let actual = read(marked), !actual.isEmpty {
                return actual == text ? .marked(marked) : .copyOnly
            }
            // Some IMK proxies expose mark and selection but not attributedSubstring.
            if start == marked.location, selected.length == 0,
               selected.location == NSMaxRange(marked) { return .marked(marked) }
        }
        if marked.location == NSNotFound || marked.length == 0, let start,
           start >= 0, start < NSNotFound - length, selected.length == 0,
           selected.location == start + length {
            let range = NSRange(location: start, length: length)
            if read(range) == text { return .committed(range) }
        }
        return .copyOnly
    }
}
