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
        let payload: [String: Any] = [
            "model": "meow-lite",
            "messages": [["role": "user", "content": trimmed]],
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
        player.play()
        players[messageID] = player // keep alive while playing
    }

    private func update(_ id: UUID, _ mutate: (inout ChatMessage) -> Void) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        mutate(&messages[index])
    }
}

private struct BubbleView: View {
    let message: ChatMessage
    let onReplay: () -> Void

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(bubbleColor)
                    .foregroundStyle(message.role == .error ? Color.white : Color.primary)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
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
                            BubbleView(message: message) {
                                model.play(message.id)
                            }
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
