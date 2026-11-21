package com.openminis.app.provider.gemini

/** A support's indices address all grounding chunks in stream order, including non-web entries. */
internal class GeminiGrounding {
    val chunkURLs = mutableListOf<String?>()
    val supports = mutableListOf<Pair<String, List<Int>>>()

    fun sourceText(answer: String, sources: Set<String>): String {
        val cited = mutableSetOf<String>()
        val lines = linkedSetOf<String>()
        for ((quote, indices) in supports) {
            if (quote.isBlank() || !answer.contains(quote)) continue
            val urls = indices.mapNotNull { chunkURLs.getOrNull(it) }.filter { it in sources }.distinct()
            if (urls.isEmpty()) continue
            cited.addAll(urls)
            val label = quote.trim().replace(Regex("""\s+"""), " ").map {
                if (it.code in 33..47 || it.code in 58..64 || it.code in 91..96 || it.code in 123..126) "\\$it" else it.toString()
            }.joinToString("")
            val links = urls.joinToString(" ") { "[${sources.indexOf(it) + 1}](<$it>)" }
            lines.add("- $label $links")
        }
        for (url in sources - cited) lines.add("- <$url>")
        return if (lines.isEmpty()) "" else lines.joinToString("\n", prefix = "\n\nSources\n")
    }
}
