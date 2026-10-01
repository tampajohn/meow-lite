// MeowLite — a minimal native macOS chat client for the meow-lite servers.
// Single-file SwiftUI app: v4/v5/v6 engine picker, chat bubbles, optional
// audio-respond mode (POST /v1/meow/audio + AVAudioPlayer replay), health
// polling. Foundation + SwiftUI + AVFoundation only.

import AVFoundation
import SwiftUI

private struct Engine: Identifiable, Hashable {
    let id: String
    let port: Int
    var label: String { id }
    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)/v1")! }
    var healthURL: URL { URL(string: "http://127.0.0.1:\(port)/health")! }
}

private let engines: [Engine] = [
    Engine(id: "v4", port: 8011), // classic
    Engine(id: "v5", port: 8012), // chaotic
    Engine(id: "v6", port: 8013), // calmer
]

private struct ChatMessage: Identifiable {
    enum Role { case user, cat, error }

    let id = UUID()
    let role: Role
    var text: String
    var audioData: Data?
    var pending: Bool
}

private struct ChatResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable { let content: String }
        let message: Message
    }
    let choices: [Choice]
}

private struct HealthResponse: Decodable {
    let status: String
    let engine: String
}

@MainActor
private final class ChatModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var audioRespond = false
    @Published var healthUp = false
    @Published var healthText = "checking…"

    /// Strong refs to in-flight players so audio isn't deallocated mid-play.
    private var players: [UUID: AVAudioPlayer] = [:]
    /// Per-session audio-variety counter (server salts the render seed with it).
    private var varietyCounter = 0
    private let session: URLSession

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        session = URLSession(configuration: config)
    }

    func send(_ prompt: String, engine: Engine) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let pending = ChatMessage(role: .cat, text: "…", audioData: nil, pending: true)
        messages.append(ChatMessage(role: .user, text: trimmed, audioData: nil, pending: false))
        messages.append(pending)
        varietyCounter += 1
        let payload: [String: Any] = [
            "model": "meow-lite",
            "messages": [["role": "user", "content": trimmed]],
            // Fresh performance per ask: same text, different cat concerto.
            // Text stays deterministic; variety only salts the audio render.
            "variety": String(varietyCounter),
        ]
        Task { await complete(pendingID: pending.id, engine: engine, payload: payload) }
    }

    private func complete(pendingID: UUID, engine: Engine, payload: [String: Any]) async {
        do {
            let text = try await chatText(engine: engine, payload: payload)
            update(pendingID) { message in
                message.text = text
                message.pending = false
            }
            guard audioRespond else { return }
            let (data, response) = try await post(
                engine.baseURL.appending(path: "meow/audio"), payload: payload
            )
            // Prefer the server-echoed text when present; both should agree.
            let headerText = (response as? HTTPURLResponse)?
                .value(forHTTPHeaderField: "X-Meow-Text")
            update(pendingID) { message in
                if let headerText, !headerText.isEmpty { message.text = headerText }
                message.audioData = data
            }
            play(pendingID)  // auto-play on receipt; replay button remains for later
        } catch {
            messages.removeAll { $0.id == pendingID }
            messages.append(
                ChatMessage(
                    role: .error,
                    text: "error: \(error.localizedDescription)",
                    audioData: nil,
                    pending: false
                )
            )
        }
    }

    private func chatText(engine: Engine, payload: [String: Any]) async throws -> String {
        let (data, _) = try await post(engine.baseURL.appending(path: "chat/completions"), payload: payload)
        let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
        guard let content = decoded.choices.first?.message.content,
              !content.isEmpty
        else {
            throw NSError(
                domain: "meowlite", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "empty response from engine"]
            )
        }
        return content
    }

    private func post(_ url: URL, payload: [String: Any]) async throws -> (Data, URLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(
                domain: "meowlite", code: code,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(code) from \(url.path)"]
            )
        }
        return (data, response)
    }

    func pollHealth(_ engine: Engine) async {
        do {
            let (data, _) = try await session.data(from: engine.healthURL)
            let health = try JSONDecoder().decode(HealthResponse.self, from: data)
            healthUp = health.status == "ok"
            healthText = "engine \(health.engine) · \(health.status) @ 127.0.0.1:\(engine.port)"
        } catch {
            healthUp = false
            healthText = "engine \(engine.id) · unreachable @ 127.0.0.1:\(engine.port)"
        }
    }

    func play(_ messageID: UUID) {
        guard let message = messages.first(where: { $0.id == messageID }),
              let data = message.audioData
        else { return }
        var player = try? AVAudioPlayer(data: data)
        if player == nil {
            // Data-init can be finicky; fall back to a temp file.
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("meowlite-\(messageID.uuidString).wav")
            try? data.write(to: url)
            player = try? AVAudioPlayer(contentsOf: url)
        }
        guard let player else {
            NSLog("MeowLite: failed to create player for message %@", messageID.uuidString)
            return
        }
        // Idempotent: stop any in-flight player for this message (no overlaps).
        players[messageID]?.stop()
        player.play()
        players[messageID] = player // keep alive while playing
    }

    /// Chip tap: fetch ONE clip for an action token and play it.
    /// Errors are silent by design — the chip just doesn't play.
    func playClip(_ action: String, messageID: UUID, engine: Engine) {
        Task { await fetchAndPlayClip(action: action, messageID: messageID, engine: engine) }
    }

    private func fetchAndPlayClip(action: String, messageID: UUID, engine: Engine) async {
        var components = URLComponents(
            url: engine.baseURL.appending(path: "meow/clip"), resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "token", value: action),
            URLQueryItem(name: "variety", value: messageID.uuidString), // message-scoped
        ]
        guard let url = components?.url else { return }
        guard let (data, response) = try? await session.data(for: URLRequest(url: url)),
              let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode),
              let player = makePlayer(data: data, messageID: messageID)
        else { return }
        // Idempotent, same rules as play(_:): no overlapping audio per message.
        players[messageID]?.stop()
        player.play()
        players[messageID] = player // strong ref while playing
    }

    private func update(_ id: UUID, _ mutate: (inout ChatMessage) -> Void) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        mutate(&messages[index])
    }

    private func makePlayer(data: Data, messageID: UUID) -> AVAudioPlayer? {
        if let player = try? AVAudioPlayer(data: data) { return player }
        // Data-init can be finicky; fall back to a temp file.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("meowlite-clip-\(messageID.uuidString).wav")
        try? data.write(to: url)
        return try? AVAudioPlayer(contentsOf: url)
    }
}

// ---------------------------------------------------------------------------
// Action-token chips
// ---------------------------------------------------------------------------

private struct ChipSpec {
    let label: String
    let symbol: String
    let color: Color
    let subtext: String
}

private let chipSpecs: [String: ChipSpec] = [
    "bite": ChipSpec(label: "BITE", symbol: "mouth", color: .red,
                     subtext: "you touched the belly 0.05s too long"),
    "scratch": ChipSpec(label: "SCRATCH", symbol: "hand.raised.fill", color: .indigo,
                        subtext: "generic scratching"),
    "scratch_couch": ChipSpec(label: "COUCH", symbol: "sofa.fill", color: .brown,
                              subtext: "the couch, specifically"),
    "knock_glass": ChipSpec(label: "GLASS", symbol: "wineglass.fill", color: .blue,
                            subtext: "it was on the table; now it's on the floor"),
    "hiss": ChipSpec(label: "HISS", symbol: "wind", color: .orange,
                     subtext: "the vet, a vacuum, or betrayal"),
    "zoomies": ChipSpec(label: "ZOOMIES", symbol: "bolt.fill", color: .purple,
                        subtext: "it is 3am. run."),
    "hairball": ChipSpec(label: "HAIRBALL", symbol: "circle.fill", color: .yellow,
                         subtext: "a gift, deposited"),
    "stare": ChipSpec(label: "STARE", symbol: "eye.fill", color: .gray,
                      subtext: "unmoving. judging."),
    "pounce": ChipSpec(label: "POUNCE", symbol: "arrow.up.right", color: .green,
                       subtext: "the red dot must die"),
    "purr": ChipSpec(label: "PURR", symbol: "heart.fill", color: .pink,
                     subtext: "you did something right"),
]

private struct Segment: Identifiable {
    let id: Int
    let text: String?
    let action: String?
}

private let actionRegex = try! NSRegularExpression(pattern: "<([A-Za-z_]+)>")

private func parseSegments(_ text: String) -> [Segment] {
    let nsText = text as NSString
    var segments: [Segment] = []
    var cursor = text.startIndex
    for match in actionRegex.matches(
        in: text, range: NSRange(location: 0, length: nsText.length)
    ) {
        guard let whole = Range(match.range, in: text),
              let nameRange = Range(match.range(at: 1), in: text)
        else { continue }
        if whole.lowerBound > cursor {
            segments.append(
                Segment(id: segments.count,
                        text: String(text[cursor ..< whole.lowerBound]),
                        action: nil)
            )
        }
        let name = String(text[nameRange]).lowercased()
        if chipSpecs[name] != nil {
            segments.append(Segment(id: segments.count, text: nil, action: name))
        } else { // not a known action: keep the literal
            segments.append(Segment(id: segments.count, text: String(text[whole]), action: nil))
        }
        cursor = whole.upperBound
    }
    if cursor < text.endIndex {
        segments.append(Segment(id: segments.count, text: String(text[cursor...]), action: nil))
    }
    return segments
}

/// Minimal wrapping flow layout (WrappingHStack-style): lays subviews in rows,
/// breaking when the next subview would exceed the available width. Text runs
/// wider than the row get a width-capped proposal so they wrap naturally;
/// chips stay .fixedSize.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    private func sizes(for subviews: Subviews, maxWidth: CGFloat) -> [CGSize] {
        subviews.map { sub in
            let ideal = sub.sizeThatFits(.unspecified)
            guard ideal.width > maxWidth else { return ideal }
            return sub.sizeThatFits(ProposedViewSize(width: maxWidth, height: nil))
        }
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let sizes = sizes(for: subviews, maxWidth: maxWidth)
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for size in sizes {
            if x > 0, x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        let width = proposal.width ?? max(0, x - (sizes.isEmpty ? 0 : spacing))
        return CGSize(width: width, height: y + rowHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        let sizes = sizes(for: subviews, maxWidth: bounds.width)
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0
        for (index, subview) in subviews.enumerated() {
            let size = sizes[index]
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

private struct ActionChipView: View {
    let spec: ChipSpec
    let onTap: () -> Void
    @State private var pressed = false

    var body: some View {
        Button {
            onTap()
            pressed = true
            Task { // brief pressed state
                try? await Task.sleep(nanoseconds: 200_000_000)
                pressed = false
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: spec.symbol)
                    .font(.callout)
                VStack(alignment: .leading, spacing: 1) {
                    Text(spec.label)
                        .font(.caption.weight(.bold))
                        .kerning(0.5)
                    Text(spec.subtext)
                        .font(.caption2)
                        .opacity(0.7)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(spec.color.opacity(pressed ? 0.85 : 0.45))
            .foregroundStyle(Color.primary)
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .fixedSize()
        .help(spec.subtext)
    }
}

private struct BubbleView: View {
    let message: ChatMessage
    let onReplay: () -> Void
    var onChipTap: ((String) -> Void)? = nil

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                bubbleContent
                if message.audioData != nil {
                    Button(action: onReplay) {
                        Label("replay", systemImage: "waveform.play")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("replay meow audio")
                }
            }
            if message.role != .user { Spacer(minLength: 48) }
        }
        .padding(.horizontal, 12)
    }

    /// Action tokens render as tappable chips inline with the text runs;
    /// errors stay plain text.
    @ViewBuilder
    private var bubbleContent: some View {
        if message.role == .error {
            Text(message.text)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(bubbleColor)
                .foregroundStyle(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 14))
        } else {
            FlowLayout(spacing: 6) {
                ForEach(parseSegments(message.text)) { segment in
                    if let action = segment.action, let spec = chipSpecs[action] {
                        ActionChipView(spec: spec) {
                            onChipTap?(action)
                        }
                    } else if let text = segment.text, !text.isEmpty {
                        Text(text)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(bubbleColor)
            .foregroundStyle(Color.primary)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var bubbleColor: Color {
        switch message.role {
        case .user: return Color.gray.opacity(0.28)
        case .cat: return Color(red: 1.0, green: 0.78, blue: 0.33).opacity(0.85) // amber-ish
        case .error: return Color.red.opacity(0.85)
        }
    }
}

private struct SplashView: View {
    var body: some View {
        ZStack {
            Color(red: 0.102, green: 0.102, blue: 0.102) // charcoal #1a1a1a
            VStack(spacing: 18) {
                if let logo = loadLogo() {
                    Image(nsImage: logo)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 240, height: 240)
                }
                Text("meow-lite")
                    .font(.system(size: 34, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.92))
            }
        }
        .ignoresSafeArea()
    }

    private func loadLogo() -> NSImage? {
        guard let path = Bundle.main.path(forResource: "logo", ofType: "png") else {
            return nil // logo not bundled; splash still shows the name
        }
        return NSImage(contentsOfFile: path)
    }
}

struct ContentView: View {
    @AppStorage("meowlite.engine") private var engineID: String = "v4"
    @StateObject private var model = ChatModel()
    @State private var draft = ""
    @State private var splashVisible = true
    @FocusState private var inputFocused: Bool

    private var engine: Engine {
        engines.first { $0.id == engineID } ?? engines[0]
    }

    var body: some View {
        ZStack {
            chatContent
                .opacity(splashVisible ? 0 : 1)
            if splashVisible {
                SplashView()
                    .transition(.opacity)
                    .zIndex(1)
            }
        }
        .task {
            // Branded splash for ~1.2 s, then crossfade into the chat UI.
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            withAnimation(.easeInOut(duration: 0.35)) {
                splashVisible = false
            }
        }
    }

    private var chatContent: some View {
        VStack(spacing: 8) {
            Picker("Engine", selection: $engineID) {
                ForEach(engines) { engine in
                    Text(engine.label).tag(engine.id)
                }
            }
            .pickerStyle(.segmented)
            .padding([.horizontal, .top], 12)

            Toggle("audio respond", isOn: $model.audioRespond)
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)

            ScrollView {
                ScrollViewReader { proxy in
                    LazyVStack(spacing: 10) {
                        ForEach(model.messages) { message in
                            BubbleView(
                                message: message,
                                onReplay: { model.play(message.id) },
                                onChipTap: { action in
                                    model.playClip(action, messageID: message.id, engine: engine)
                                }
                            )
                            .id(message.id)
                        }
                    }
                    .padding(.vertical, 4)
                    .onChange(of: model.messages.count) { _, _ in
                        if let last = model.messages.last?.id {
                            withAnimation(.easeOut(duration: 0.15)) {
                                proxy.scrollTo(last, anchor: .bottom)
                            }
                        }
                    }
                }
            }

            HStack(spacing: 8) {
                TextField("say something…", text: $draft)
                    .focused($inputFocused)
                    .onSubmit(send)
                    .textFieldStyle(.roundedBorder)
                Button("Send", action: send)
                    .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(.horizontal, 12)

            HStack(spacing: 6) {
                Circle()
                    .fill(model.healthUp ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(model.healthText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding([.horizontal, .bottom], 14)
        }
        .frame(minWidth: 420, minHeight: 560)
        .task(id: engine.id) {
            while !Task.isCancelled {
                await model.pollHealth(engine)
                try? await Task.sleep(nanoseconds: 10_000_000_000) // every 10 s
            }
        }
    }

    private func send() {
        let prompt = draft
        draft = ""
        inputFocused = true
        model.send(prompt, engine: engine)
    }
}

@main
struct MeowLiteApp: App {
    var body: some Scene {
        WindowGroup("meow-lite") {
            ContentView()
        }
        .windowResizability(.contentMinSize)
    }
}
