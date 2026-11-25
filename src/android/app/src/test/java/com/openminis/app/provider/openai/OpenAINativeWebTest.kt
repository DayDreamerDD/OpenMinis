package com.openminis.app.provider.openai

import com.openminis.app.data.model.AgentToolDefinition
import com.openminis.app.data.model.LLMMessage
import com.openminis.app.data.model.LLMModel
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenAINativeWebTest {
    private fun types(id: String, responses: Boolean, enabled: Boolean = true,
                      base: String = "https://api.deepseek.com"): List<String> {
        val provider = OpenAIProvider(apiKey = "k", model = LLMModel(id = id, displayName = id, provider = "Test"), basePath = base)
        val history = listOf(LLMMessage(LLMMessage.Role.USER, "hello"))
        val tools = listOf(AgentToolDefinition("local_tool", "", emptyMap()))
        val body = if (responses) {
            provider.buildResponsesAPIBody(history, null, 1024, true, tools = tools, nativeWebEnabled = enabled)
        } else {
            provider.buildRequestBody(history, null, 1024, true, null, emptyList(), tools, nativeWebEnabled = enabled)
        }
        val wireTools = body.getJSONArray("tools")
        return (0 until wireTools.length()).map { wireTools.getJSONObject(it).getString("type") }
    }

    @Test fun modelProtocolAndOptIn() {
        for (responses in listOf(true, false)) {
            assertEquals(listOf("function", "deepseek_web_search", "deepseek_read_url"), types("deepseek-flash", responses))
            assertEquals(listOf("function"), types("deepseek-flash", responses, false))
            assertEquals(listOf("function"), types("unknown", responses))
        }
        assertEquals(listOf("function", "web_search"), types("deepseek-v4-pro", true))
        assertEquals(listOf("function"), types("deepseek-v4-pro", false))
        assertEquals(listOf("function"), types("deepseek-flash", false, base = "https://api.deepseek.com.example.com"))
        assertEquals(listOf("function", "web_search"), types("relay-model", true, base = "https://example.com"))
    }

    @Test fun webCallHistoryStaysWithItsOrigin() {
        val endpoint = "https://api.deepseek.com"
        val model = LLMModel(id = "deepseek-flash", displayName = "Flash", provider = "Test")
        val provider = OpenAIProvider(apiKey = "k", model = model, basePath = endpoint)
        val call = JSONObject().put("type", "web_search_call").put("signature", "opaque")
        val envelope = JSONObject().put("items", JSONArray().put(call))
        fun input(responses: Boolean, enabled: Boolean): String {
            val history = listOf(
                LLMMessage(LLMMessage.Role.ASSISTANT, "answer", nativeWebReplay = envelope.toString()),
                LLMMessage(LLMMessage.Role.USER, "follow up"),
            )
            return if (responses) {
                provider.buildResponsesAPIBody(history, null, 1024, true, nativeWebEnabled = enabled).getJSONArray("input").toString()
            } else {
                provider.buildRequestBody(history, null, 1024, true, null, emptyList(), nativeWebEnabled = enabled)
                    .getJSONArray("messages").toString()
            }
        }
        for (responses in listOf(false, true)) {
            envelope.put("base", endpoint).put("model", model.id).put("responses", responses)
            val replayed = input(responses, true)
            assertTrue("opaque" in replayed)
            assertTrue("answer" in replayed)
            assertFalse("opaque" in input(!responses, true))
            assertFalse("opaque" in input(responses, false))
            envelope.put("model", "another-model")
            assertFalse("opaque" in input(responses, true))
            envelope.put("model", model.id).put("base", "https://other.example.com")
            assertFalse("opaque" in input(responses, true))
        }
    }
}
