package com.openminis.app.provider.gemini

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiNativeWebTest {
    @Test fun signedHistoryRequiresIdenticalTextAndToolArguments() {
        val parts = """[{"thought":true,"text":"private"},{"toolCall":{"id":"search"}},{"text":"臺北 🌧"},{"functionCall":{"name":"write","args":{"flag":true,"where":{"b":"note","a":2}}}}]"""
        val args = """{"where":{"a":2,"b":"note"},"flag":true}"""
        val current = listOf("write" to args)
        val text = "臺北 🌧\nSources"
        assertTrue(GeminiWebReplay.matches(parts, "\nSources", text, current))
        assertFalse(GeminiWebReplay.matches(parts, "\nSources", "edited", current))
        assertFalse(GeminiWebReplay.matches(parts, "\nSources", text, emptyList()))
        assertFalse(GeminiWebReplay.matches(parts, "\nSources", text, listOf("write" to """{"where":"[pruned]"}""")))
        assertFalse(GeminiWebReplay.matches(parts, "\nSources", text, listOf("write" to args.replace("true", "1"))))
    }

    @Test fun lateGroundingUsesGlobalChunkIndicesAndPreservesUnattributedSources() {
        val grounding = GeminiGrounding()
        val weather = "https://example.com/weather"
        val other = "https://example.com/page"
        grounding.chunkURLs.addAll(listOf(null, weather, other))
        grounding.supports.add("臺北 [rain]" to listOf(0, 1, 1, 9))
        grounding.supports.add("absent quote" to listOf(2))
        val footer = grounding.sourceText("臺北 [rain] 🌧", linkedSetOf(weather, other))
        assertEquals("\n\nSources\n- 臺北 \\[rain\\] [1](<$weather>)\n- <$other>", footer)
    }
}
