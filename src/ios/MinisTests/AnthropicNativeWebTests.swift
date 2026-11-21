import XCTest
@testable import Minis

final class AnthropicNativeWebTests: XCTestCase {
    func testToolDialectAndModelGate() {
        let base = "https://api.deepseek.com/anthropic"
        XCTAssertEqual(AnthropicWebTools.declarations(base: base, model: "deepseek-flash").compactMap { $0["type"] as? String },
                       ["deepseek_web_search", "deepseek_read_url"])
        XCTAssertTrue(AnthropicWebTools.declarations(base: base, model: "deepseek-pro").isEmpty)
        let claude = AnthropicWebTools.declarations(base: "https://api.anthropic.com", model: "claude-sonnet-5")
        XCTAssertEqual(claude.compactMap { $0["type"] as? String }, ["web_search_20250305", "web_fetch_20250910"])
        XCTAssertEqual((claude.last?["citations"] as? [String: Bool])?["enabled"], true)
    }

    func testMixedAssistantHistoryRestoresOnlyMarkedSpan() {
        let blocks: [[String: Any]] = [
            ["type": "thinking", "thinking": "signed", "signature": "signature"],
            ["type": "server_tool_use", "id": "server", "name": "web_search"],
            ["type": "tool_use", "id": "client", "name": "local", "input": [:]]
        ]
        let marker = ["type": "text", "text": "request-marker"]
        var history: [[String: Any]] = [["role": "assistant", "content": [
            ["type": "thinking", "thinking": "synthesized"],
            ["type": "text", "text": "before"], marker, ["type": "text", "text": "rendered source link"],
            marker, ["type": "text", "text": "after"], ["type": "tool_use", "id": "client", "name": "local", "input": [:]]
        ]]]
        AnthropicWebTools.restoreHistory(&history, replay: ["request-marker": blocks])
        let replay = history[0]["content"] as? [[String: Any]] ?? []
        XCTAssertEqual(replay.compactMap { $0["type"] as? String }, ["text", "thinking", "server_tool_use", "tool_use", "text"])
        XCTAssertEqual(replay[1]["signature"] as? String, "signature")
    }
}

private final class AnthropicNativeWebFixture: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let data: Data
        if let body = request.httpBody {
            data = body
        } else if let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var body = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while true {
                let n = stream.read(&buffer, maxLength: buffer.count)
                if n <= 0 { break }
                body.append(contentsOf: buffer.prefix(n))
            }
            data = body
        } else { data = Data() }
        do {
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let messages = try XCTUnwrap(body["messages"] as? [[String: Any]])
            let tools = try XCTUnwrap(body["tools"] as? [[String: Any]])
            XCTAssertEqual(tools.compactMap { $0["type"] as? String }, ["web_search_20250305", "web_fetch_20250910"])
            let frames: [String]
            if messages.count == 1 {
                frames = [
                    #"{"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
                    #"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv","name":"web_search","input":{}}}"#,
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"news\"}"}}"#,
                    #"{"type":"content_block_stop","index":0}"#,
                    #"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srv","content":[{"encrypted_content":"signature"}]}}"#,
                    #"{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}"#,
                    #"{"type":"message_stop"}"#
                ]
            } else {
                let blocks = try XCTUnwrap(messages.last?["content"] as? [[String: Any]])
                XCTAssertEqual((blocks[0]["input"] as? [String: String])?["query"], "news")
                XCTAssertEqual((blocks[1]["content"] as? [[String: String]])?.first?["encrypted_content"], "signature")
                frames = [
                    #"{"type":"message_start","message":{"usage":{"input_tokens":12}}}"#,
                    #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}"#,
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"url":"https://example.com/source"}}}"#,
                    #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
                    #"{"type":"message_stop"}"#
                ]
            }
            let reply = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                        headerFields: ["Content-Type": "text/event-stream"])!
            client?.urlProtocol(self, didReceive: reply, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(frames.map { "data: \($0)\n\n" }.joined().utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

extension AnthropicNativeWebTests {
    func testNativeStreamResumesAndReturnsCitedAnswer() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [AnthropicNativeWebFixture.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        var model = LLMModel.claudeHaiku45
        model.nativeWebEnabled = true
        let agent = AnthropicAgentProvider(provider: AnthropicProvider(apiKey: "fixture", model: model, nativeWebSession: session))
        let stream = try await agent.streamAgentMessageClamped(messages: [AgentMessage(role: .user, parts: [.text("News?")])],
            systemPrompt: nil, tools: [], maxTokens: 256, thinkingLevel: .off)
        var text = ""
        var replay: String?
        var usage: LLMUsage?
        var done = false
        for try await event in stream {
            switch event {
            case .textDelta(let fragment): text += fragment
            case .reasoningEcho(let echo): replay = echo.nativeWebReplay
            case .usage(let value): usage = value
            case .done(.endTurn): done = true
            case .toolCallComplete: XCTFail("A server tool was dispatched to the client")
            default: break
            }
        }
        XCTAssertEqual(text, "Answer <https://example.com/source>")
        XCTAssertTrue(replay?.contains("signature") == true)
        XCTAssertEqual(usage?.inputTokens, 22)
        XCTAssertEqual(usage?.outputTokens, 7)
        XCTAssertTrue(done)
    }
}
