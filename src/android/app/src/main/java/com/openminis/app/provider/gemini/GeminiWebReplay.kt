package com.openminis.app.provider.gemini

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Replay signed parts only while their user-visible projection still matches the app's history. */
internal object GeminiWebReplay {
    fun matches(parts: String, sourceText: String, currentText: String, calls: List<Pair<String, String>>): Boolean =
        runCatching {
            val raw = Json.parseToJsonElement(parts).jsonArray.map { it.jsonObject }
            val text = raw.filter { it["thought"]?.jsonPrimitive?.booleanOrNull != true }
                .mapNotNull { it["text"]?.jsonPrimitive?.content }.joinToString("")
            val functions = raw.mapNotNull { it["functionCall"]?.jsonObject }
            text + sourceText == currentText && functions.size == calls.size &&
                functions.zip(calls).all { (function, call) ->
                    function["name"]?.jsonPrimitive?.content == call.first &&
                        (function["args"] ?: JsonObject(emptyMap())) == Json.parseToJsonElement(call.second)
                }
        }.getOrDefault(false)
}
