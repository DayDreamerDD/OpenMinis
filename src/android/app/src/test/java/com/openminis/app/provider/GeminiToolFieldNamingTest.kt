package com.openminis.app.provider

import com.openminis.app.data.model.AgentContentPart
import com.openminis.app.data.model.AgentToolDefinition
import com.openminis.app.data.model.AgentToolParam
import com.openminis.app.data.model.LLMMessage
import com.openminis.app.data.model.LLMStreamChunk
import com.openminis.app.data.model.LLMModel
import com.openminis.app.provider.gemini.GeminiProvider
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * [T-android-gemini-tool-field] The Gemini tool wrapper must be the REST
 * spelling, `functionDeclarations`, on every request path.
 *
 * `function_declarations` is the proto/gRPC field name. Google's own endpoint
 * accepts both, so this was invisible on a direct key — but a relay that
 * transcodes the request does not: against the same Flash model, camelCase
 * returned a real `functionCall` while snake_case returned TOOL_UNAVAILABLE,
 * and a native streaming request answered HTTP 200 with `parts: []` and
 * finishReason STOP, which surfaces to the user as "the model returned an
 * empty response". iOS has always sent camelCase, which is why the same
 * account worked there and not here.
 *
 * These assert on the bytes actually put on the wire, through the real
 * provider, because the bug was invisible at every other layer: the tool
 * schema, the response parser and the history replay were all already correct.
 */
class GeminiToolFieldNamingTest {

    private lateinit var server: MockWebServer

    @Before fun setUp() { server = MockWebServer(); server.start() }
    @After fun tearDown() { server.shutdown() }

    private fun provider() = GeminiProvider(
        apiKey = "test-key",
        model = LLMModel.gemini25Flash,
        basePath = server.url("/").toString().trimEnd('/'),
    )

    private fun tools() = listOf(
        AgentToolDefinition(
            name = "file_write",
            description = "Write a file",
            parameters = mapOf("path" to AgentToolParam("string", "Where to write")),
            required = listOf("path"),
        ),
    )

    private fun assertCamelCaseToolWrapper(body: JSONObject) {
        val toolsArr = body.optJSONArray("tools")
        assertTrue("request must carry a tools array: $body", toolsArr != null && toolsArr.length() == 1)
        val wrapper = toolsArr!!.getJSONObject(0)
        assertTrue(
            "tool wrapper must be `functionDeclarations` (REST spelling): $wrapper",
            wrapper.has("functionDeclarations"),
        )
        assertFalse(
            "`function_declarations` is the proto name and is rejected by relays: $wrapper",
            wrapper.has("function_declarations"),
        )
        assertEquals(
            "the declaration itself must survive the rename",
            "file_write",
            wrapper.getJSONArray("functionDeclarations").getJSONObject(0).getString("name"),
        )
        // iOS pairs the declarations with an explicit AUTO mode on both paths;
        // leaving the mode to the endpoint's default lets a relay decline to
        // call tools at all while still answering 200.
        assertEquals(
            "toolConfig must request AUTO function calling, matching iOS: $body",
            "AUTO",
            body.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode"),
        )
    }

    @Test
    fun `streaming request declares tools in camelCase with an AUTO toolConfig`() = runBlocking {
        server.enqueue(
            MockResponse()
                .setHeader("Content-Type", "text/event-stream")
                .setBody(
                    "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]}," +
                        "\"finishReason\":\"STOP\"}]}\n\n",
                ),
        )
        provider().streamMessage(
            messages = listOf(LLMMessage(LLMMessage.Role.USER, "write a file")),
            systemPrompt = null,
            maxTokens = 1024,
            tools = tools(),
        ).toList()

        assertCamelCaseToolWrapper(JSONObject(server.takeRequest().body.readUtf8()))
    }

    @Test
    fun `non-streaming request declares tools in camelCase with an AUTO toolConfig`() = runBlocking {
        server.enqueue(
            MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody(
                    """{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},""" +
                        """"finishReason":"STOP"}]}""",
                ),
        )
        provider().sendMessage(
            messages = listOf(LLMMessage(LLMMessage.Role.USER, "write a file")),
            systemPrompt = null,
            maxTokens = 1024,
            tools = tools(),
        )

        assertCamelCaseToolWrapper(JSONObject(server.takeRequest().body.readUtf8()))
    }

    @Test
    fun `a request without tools carries neither tools nor toolConfig`() = runBlocking {
        // toolConfig must not leak onto plain chat turns: a config naming a
        // calling mode with no declarations is a shape the endpoint has no
        // reason to accept.
        server.enqueue(
            MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody(
                    """{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},""" +
                        """"finishReason":"STOP"}]}""",
                ),
        )
        provider().sendMessage(
            messages = listOf(LLMMessage(LLMMessage.Role.USER, "hello")),
            systemPrompt = null,
            maxTokens = 1024,
        )

        val body = JSONObject(server.takeRequest().body.readUtf8())
        assertFalse("no tools key on a toolless turn: $body", body.has("tools"))
        assertFalse("no toolConfig key on a toolless turn: $body", body.has("toolConfig"))
    }
    @Test
    fun `native web keeps custom tools and collects metadata-only sources`() = runBlocking {
        val mixed = GeminiProvider("key", LLMModel.gemini25Flash.copy(id = "gemini-3.8-flash"),
            server.url("/").toString().trimEnd('/'))
        server.enqueue(MockResponse().setBody(
            """data: {"candidates":[{"content":{"parts":[{"toolCall":{"id":"a","toolType":"GOOGLE_SEARCH_WEB"},"thoughtSignature":"signed"},{"text":"Answer"},{"functionCall":{"id":"wire-call","name":"file_write","args":{"path":"note"}}}]}}]}""" + "\n\n" +
            """data: {"candidates":[{"finishReason":"STOP","groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com"}},{"web":{"uri":"javascript:alert(1)"}}]},"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}}]}""" + "\n\n"
        ))
        val chunks = mixed.streamMessage(messages = listOf(LLMMessage(LLMMessage.Role.USER, "look up")),
            systemPrompt = null, maxTokens = 1024, tools = tools(), nativeWebEnabled = true).toList()
        val request = JSONObject(server.takeRequest().body.readUtf8())
        val declarations = request.getJSONArray("tools")
        assertTrue(declarations.getJSONObject(0).has("googleSearch"))
        assertTrue(declarations.getJSONObject(1).has("urlContext"))
        assertTrue(declarations.getJSONObject(2).has("functionDeclarations"))
        val config = request.getJSONObject("toolConfig")
        assertTrue(config.getBoolean("includeServerSideToolInvocations"))
        assertFalse(config.has("functionCallingConfig"))
        val text = chunks.filterIsInstance<LLMStreamChunk.Text>().joinToString("") { it.text }
        assertEquals("Answer\n\nSources\n- <https://example.com>", text)
        val replay = JSONObject(chunks.filterIsInstance<LLMStreamChunk.Finished>().single().nativeWebReplay!!)
        assertEquals("signed", replay.getJSONArray("parts").getJSONObject(0).getString("thoughtSignature"))

        val tool = chunks.filterIsInstance<LLMStreamChunk.ToolCallComplete>().single()
        val history = listOf(
            LLMMessage(LLMMessage.Role.USER, "look up"),
            LLMMessage(LLMMessage.Role.ASSISTANT, "", nativeWebReplay = replay.toString(),
                contentParts = listOf(AgentContentPart.Text(text), AgentContentPart.ToolUse(tool.id, "file_write", JSONObject().put("path", "note")))),
            LLMMessage(LLMMessage.Role.USER, "", contentParts = listOf(AgentContentPart.ToolResult(tool.id, "file_write", "ok"))),
        )
        server.enqueue(MockResponse().setBody(
            """data: {"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}""" + "\n\n"
        ))
        mixed.streamMessage(history, null, 1024, tools = tools(), nativeWebEnabled = true).toList()
        val contents = JSONObject(server.takeRequest().body.readUtf8()).getJSONArray("contents")
        assertEquals(replay.getJSONArray("parts").toString(), contents.getJSONObject(1).getJSONArray("parts").toString())
        val result = contents.getJSONObject(2).getJSONArray("parts").getJSONObject(0).getJSONObject("functionResponse")
        assertEquals("wire-call", result.getString("id"))
    }

}
