package com.openminis.app.provider

import com.openminis.app.data.model.LLMMessage
import com.openminis.app.data.model.LLMStreamChunk
import com.openminis.app.data.model.ThinkingLevel
import com.openminis.app.provider.anthropic.AnthropicProvider
import com.openminis.app.provider.anthropic.AnthropicWebTools
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class AnthropicNativeWebTest {
    private fun events(vararg data: String) = MockResponse().setHeader("Content-Type", "text/event-stream")
        .setBody(data.joinToString("") { "data: $it\n\n" })

    @Test fun pausePreservesSignedBlocksAndCitationsWithoutLocalDispatch() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val base = server.url("/").toString().trimEnd('/')
            val provider = AnthropicProvider(apiKey = "test", basePath = base)
            server.enqueue(events(
                """{"type":"message_start","message":{"usage":{"input_tokens":10}}}""",
                """{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","name":"web_search","id":"srv","input":{}}}""",
                """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"news\"}"}}""",
                """{"type":"content_block_stop","index":0}""",
                """{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srv","content":[{"encrypted_content":"signed"}]}}""",
                """{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3}}""",
                """{"type":"message_stop"}"""
            ))
            server.enqueue(events(
                """{"type":"message_start","message":{"usage":{"input_tokens":15}}}""",
                """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""",
                """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}""",
                """{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.com/news"}}}""",
                """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}""",
                """{"type":"message_stop"}"""
            ))
            val chunks = provider.streamMessageClamped(listOf(LLMMessage(LLMMessage.Role.USER, "News?")),
                null, 256, null, emptyList(), emptyList(), ThinkingLevel.OFF, true).toList()
            val first = JSONObject(server.takeRequest().body.readUtf8())
            assertEquals("web_search_20250305", first.getJSONArray("tools").getJSONObject(0).getString("type"))
            val paused = JSONObject(server.takeRequest().body.readUtf8()).getJSONArray("messages").getJSONObject(1)
                .getJSONArray("content")
            assertEquals("news", paused.getJSONObject(0).getJSONObject("input").getString("query"))
            assertEquals("signed", paused.getJSONObject(1).getJSONArray("content").getJSONObject(0).getString("encrypted_content"))
            assertTrue(chunks.none { it is LLMStreamChunk.ToolCallComplete })
            val usage = chunks.filterIsInstance<LLMStreamChunk.Usage>().last().usage
            assertEquals(25, usage.inputTokens)
            assertEquals(5, usage.outputTokens)
            val raw = chunks.filterIsInstance<LLMStreamChunk.Finished>().single().nativeWebReplay
            val message = LLMMessage(LLMMessage.Role.ASSISTANT, "Answer <https://example.com/news>", nativeWebReplay = raw)
            val replay = AnthropicWebTools.restore(message, base, provider.model.id)!!
            assertEquals("https://example.com/news", replay.getJSONObject(2).getJSONArray("citations").getJSONObject(0).getString("url"))
            assertNull(AnthropicWebTools.restore(message, "https://other.example", provider.model.id))
            assertNull(AnthropicWebTools.restore(message.copy(content = "pruned"), base, provider.model.id))
        }
    }

    @Test fun officialDeepSeekUsesItsOwnWebDeclarations() {
        val body = JSONObject()
        val base = "https://api.deepseek.com/anthropic"
        assertFalse(AnthropicWebTools.addTo(body, base, "deepseek-pro"))
        assertTrue(AnthropicWebTools.addTo(body, base, "deepseek-flash"))
        assertEquals("deepseek_web_search", body.getJSONArray("tools").getJSONObject(0).getString("type"))
        assertEquals("deepseek_read_url", body.getJSONArray("tools").getJSONObject(1).getString("type"))
    }
}
