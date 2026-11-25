package com.openminis.app.provider.anthropic

import com.openminis.app.data.model.AgentContentPart
import com.openminis.app.data.model.LLMError
import com.openminis.app.data.model.LLMMessage
import kotlinx.serialization.json.Json
import java.net.URI
import org.json.JSONArray
import org.json.JSONObject

internal object AnthropicWebTools {
    // message_delta often carries only output_tokens. Merge within a request,
    // then add usage billed by any earlier pause_turn requests.
    fun usage(previous: JSONObject, current: JSONObject, update: JSONObject): JSONObject {
        val total = JSONObject()
        for (key in listOf("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) {
            if (update.has(key)) current.put(key, update.optInt(key))
            if (previous.has(key) || current.has(key)) total.put(key, previous.optInt(key) + current.optInt(key))
        }
        return total
    }

    fun encodeReplay(base: String, model: String, preceding: JSONArray, blocks: JSONArray, rendered: String): String {
        val all = JSONArray()
        for (items in listOf(preceding, blocks)) for (i in 0 until items.length()) all.put(items.get(i))
        return JSONObject().put("protocol", "anthropic-messages").put("base", base)
            .put("model", model).put("text", rendered).put("blocks", all).toString()
    }

    fun restore(message: LLMMessage, base: String, model: String): JSONArray? {
        val raw = message.nativeWebReplay ?: return null
        val restored = restoreOrNull(message, raw, base, model)
        if (restored == null) {
            // The turn genuinely used native web but the visible projection no
            // longer matches (edited, pruned, offloaded arguments, model/base
            // switch). Falling back to plain text is correct, but silent.
            android.util.Log.d(
                "AnthropicWebTools",
                "native web replay dropped: model=$model base=$base — history no longer matches the signed blocks",
            )
        }
        return restored
    }

    private fun restoreOrNull(message: LLMMessage, raw: String, base: String, model: String): JSONArray? {
        val obj = runCatching { JSONObject(raw) }.getOrNull() ?: return null
        if (obj.optString("protocol") != "anthropic-messages" || obj.optString("base") != base || obj.optString("model") != model) return null
        val blocks = obj.optJSONArray("blocks") ?: return null
        val parts = message.contentParts
        if (parts.any { it !is AgentContentPart.Text && it !is AgentContentPart.ToolUse }) return null
        val visible = if (parts.isEmpty()) message.content else parts.filterIsInstance<AgentContentPart.Text>().joinToString("") { it.text }
        if (visible.trim() != obj.optString("text").trim()) return null
        val calls = parts.filterIsInstance<AgentContentPart.ToolUse>()
        val original = (0 until blocks.length()).map { blocks.getJSONObject(it) }.filter { it.optString("type") == "tool_use" }
        if (calls.size != original.size) return null
        for ((call, block) in calls.zip(original)) {
            if (call.isOffloadedArgument || call.id != block.optString("id") || call.name != block.optString("name")) return null
            val input = block.optJSONObject("input") ?: return null
            if (Json.parseToJsonElement(call.input.toString()) != Json.parseToJsonElement(input.toString())) return null
        }
        return blocks
    }

    /**
     * A 400 that names the hosted web tools. Relays that speak the Messages
     * protocol often implement the chat surface but not `web_search_20250305`;
     * the caller retries once without them rather than failing the turn.
     */
    fun isUnsupportedToolError(error: Throwable): Boolean {
        val provider = error as? LLMError.ProviderError ?: return false
        if (provider.httpStatus != 400) return false
        val detail = provider.detail.lowercase()
        if ("web_search" in detail || "web_fetch" in detail || "deepseek_web" in detail) return true
        val unsupported = listOf("unsupported", "not supported", "not support", "unknown", "invalid")
        return "tool" in detail && unsupported.any { it in detail }
    }

    fun addTo(body: JSONObject, basePath: String, model: String): Boolean {
        val host = runCatching { URI(basePath).host }.getOrNull()
        val deepSeek = host.equals("api.deepseek.com", ignoreCase = true)
        if (deepSeek && model != "deepseek-flash") return false
        val tools = body.optJSONArray("tools") ?: JSONArray()
        val declarations = if (deepSeek) listOf(
            "deepseek_web_search" to "deepseek_web_search", "deepseek_read_url" to "deepseek_read_url"
        ) else listOf("web_search" to "web_search_20250305", "web_fetch" to "web_fetch_20250910")
        val names = (0 until tools.length()).map { tools.getJSONObject(it).optString("name") }
        for ((name, type) in declarations) {
            if (name in names) continue
            val tool = JSONObject().put("name", name).put("type", type)
            if (name == "web_fetch") tool.put("citations", JSONObject().put("enabled", true))
            tools.put(tool)
        }
        body.put("tools", tools)
        return true
    }
}

/** Exact assistant content required when the service pauses a native tool turn. */
internal class AnthropicWebBlocks(history: JSONArray) {
    var usedWeb = false
        private set
    val content = JSONArray()
    private val partial = StringBuilder()
    private val documents = mutableListOf<String?>()

    init {
        for (i in 0 until history.length()) {
            val blocks = history.optJSONObject(i)?.optJSONArray("content") ?: continue
            for (j in 0 until blocks.length()) blocks.optJSONObject(j)?.let(::recordDocument)
        }
    }

    private fun recordDocument(block: JSONObject) {
        when (block.optString("type")) {
            "document" -> documents.add(null)
            "web_fetch_tool_result" -> block.optJSONObject("content")?.takeIf {
                it.optJSONObject("content")?.optString("type") == "document"
            }?.let { documents.add(it.optString("url")) }
        }
    }
    private val links = mutableSetOf<String>()

    fun accept(event: JSONObject): String? {
        val index = event.optInt("index", -1)
        if (index < 0) return null
        when (event.optString("type")) {
            "content_block_start" -> {
                val block = event.getJSONObject("content_block")
                if (block.optString("type") == "server_tool_use") usedWeb = true
                content.put(index, block)
                partial.clear()
                recordDocument(block)
            }
            "content_block_delta" -> {
                val block = content.optJSONObject(index) ?: return null
                val delta = event.getJSONObject("delta")
                when (delta.optString("type")) {
                    "text_delta", "thinking_delta", "signature_delta" -> {
                        val key = delta.getString("type").removeSuffix("_delta")
                        block.put(key, block.optString(key) + delta.optString(key))
                    }
                    "input_json_delta" -> partial.append(delta.optString("partial_json"))
                    "citations_delta" -> {
                        val citation = delta.getJSONObject("citation")
                        val citations = block.optJSONArray("citations") ?: JSONArray()
                        block.put("citations", citations.put(citation))
                        val url = citation.optString("url").ifBlank {
                            documents.getOrNull(citation.optInt("document_index", -1)).orEmpty()
                        }
                        if (runCatching { URI(url).scheme?.lowercase() }.getOrNull() in setOf("https", "http") &&
                            url.none { it.isWhitespace() || it == '<' || it == '>' } && links.add(url)) return " <$url>"
                    }
                }
            }
            "content_block_stop" -> if (partial.isNotEmpty()) {
                content.getJSONObject(index).put("input", JSONObject(partial.toString()))
                partial.clear()
            }
        }
        return null
    }
}
