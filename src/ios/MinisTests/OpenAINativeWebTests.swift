import XCTest
@testable import Minis

final class OpenAINativeWebTests: XCTestCase {
    func testModelSettingsAndReplayRoundTrip() throws {
        let old = try JSONDecoder().decode(ModelOverrides.self, from: Data("{}".utf8))
        XCTAssertNil(old.nativeWebEnabled)
        let optIn = ModelOverrides(nativeWebEnabled: true)
        XCTAssertFalse(optIn.isEmpty)
        XCTAssertEqual(try JSONDecoder().decode(ModelOverrides.self, from: JSONEncoder().encode(optIn)), optIn)

        let echo = ReasoningEcho(providerKind: "deepseek-chat", modelId: "deepseek-flash",
                                 upstreamIdentity: "https://api.deepseek.com", items: [],
                                 nativeWebReplay: "[{}]")
        let restored = try XCTUnwrap(ReasoningEcho(persistedJSON: echo.persistableJSON))
        XCTAssertEqual(restored.upstreamIdentity, echo.upstreamIdentity)
        XCTAssertEqual(restored.modelId, echo.modelId)
        XCTAssertEqual(restored.nativeWebReplay, echo.nativeWebReplay)
    }

    func testProtocolsAndModels() {
        let types = OpenAIAgentProvider.nativeWebTypes
        for responses in [true, false] {
            XCTAssertEqual(types(true, responses, true, "deepseek-flash"), ["deepseek_web_search", "deepseek_read_url"])
            XCTAssertEqual(types(false, responses, true, "deepseek-flash"), [])
            XCTAssertEqual(types(true, responses, true, "unlisted"), [])
        }
        XCTAssertEqual(types(true, true, true, "deepseek-v4-pro"), ["web_search"])
        XCTAssertEqual(types(true, false, true, "deepseek-v4-pro"), [])
        XCTAssertEqual(types(true, true, false, "relay-model"), ["web_search"])
        XCTAssertEqual(types(true, false, false, "relay-model"), [])
    }

    func testCitationFooterKeepsInlineLinksAndAddsOnlyMissingSources() {
        let docs = "https://developers.openai.com/api/docs/guides/tools-web-search?api-mode=responses"
        let other = "https://developers.openai.com/guide_(tools)?a=1#intro"
        let answer = "官方文档。([developers.openai.com](\(docs)))"
        XCTAssertEqual(OpenAIAgentProvider.citationFooter(answer, [docs + "&utm_source=openai", docs]), "")
        XCTAssertEqual(OpenAIAgentProvider.citationFooter(answer, [docs, other, other]),
                       "\n\n[developers.openai.com](https://developers.openai.com/guide_%28tools%29?a=1#intro)")
        XCTAssertEqual(OpenAIAgentProvider.citationFooter("developers.openai.com", [docs]),
                       "\n\n[developers.openai.com](\(docs))")
    }

    func testWireCitationShapes() {
        let href = "https://example.com/report"
        XCTAssertEqual(OpenAIAgentProvider.citationURL(["type": "url_citation", "url": href]), href)
        XCTAssertEqual(OpenAIAgentProvider.citationURL(["type": "url_citation", "url_citation": ["url": href]]), href)
        XCTAssertNil(OpenAIAgentProvider.citationURL(["type": "url_citation", "url": "javascript:alert(1)"]))
    }

    func testCurrentMessageAndProviderStepsSurviveConversion() {
        let endpoint = "https://api.deepseek.com"
        var model = LLMModel(id: "deepseek-flash", displayName: "Flash", provider: "DeepSeek")
        model.nativeWebEnabled = true
        let agent = OpenAIAgentProvider(provider: OpenAIProvider(apiKey: "test", model: model, customBaseURL: endpoint))
        var message = AgentMessage(role: .assistant, parts: [.text("edited answer")])
        let replay = #"[{"type":"web_search_call","signature":"opaque"}]"#
        message.reasoningEcho = ReasoningEcho(providerKind: "deepseek-chat", modelId: model.id,
                                               upstreamIdentity: endpoint, items: [], nativeWebReplay: replay)
        let chat = agent.flattenChatCompletionsMessages([message])
        XCTAssertEqual((chat.first?["server_tool_calls"] as? [[String: String]])?.first?["signature"], "opaque")
        XCTAssertNil(agent.convertMessagesResponsesAPI([message]).first?["signature"])

        message.reasoningEcho = ReasoningEcho(providerKind: OpenAIAgentProvider.responsesAPIProviderKind, modelId: model.id,
                                               upstreamIdentity: endpoint, items: [], nativeWebReplay: replay)
        let responses = agent.convertMessagesResponsesAPI([message])
        XCTAssertEqual(responses.first?["signature"] as? String, "opaque")
        XCTAssertEqual(responses.last?["content"] as? String, "edited answer")
        XCTAssertNil(agent.flattenChatCompletionsMessages([message]).first?["server_tool_calls"])
    }
}
