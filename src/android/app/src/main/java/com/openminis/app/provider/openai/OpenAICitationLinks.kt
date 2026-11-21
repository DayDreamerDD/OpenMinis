package com.openminis.app.provider.openai

import java.net.URI
import java.net.URISyntaxException

private val inlineDestination = Regex("""\]\((https?://[^\s)]+)\)""", RegexOption.IGNORE_CASE)

// OpenAI sometimes adds its tracking parameter to only one of the two links.
private fun citationKey(url: URI): String = buildString {
    append(url.scheme.lowercase()).append("://").append(url.rawAuthority.lowercase())
    append(url.rawPath.ifEmpty { "/" })
    val query = url.rawQuery?.split('&')?.filterNot { it == "utm_source=openai" }
    if (!query.isNullOrEmpty()) append('?').append(query.joinToString("&"))
    url.rawFragment?.let { append('#').append(it) }
}

internal fun openAICitationFooter(text: CharSequence, urls: Collection<String>): String {
    if (urls.isEmpty()) return ""
    val linked = mutableSetOf<String>()
    fun parse(href: String): URI? = try {
        URI(href).takeIf { it.host != null }
    } catch (_: URISyntaxException) {
        null
    }
    for (match in inlineDestination.findAll(text)) {
        parse(match.groupValues[1])?.let { linked.add(citationKey(it)) }
    }
    val links = urls.mapNotNull { href ->
        val destination = href.replace("(", "%28").replace(")", "%29")
        val url = parse(destination) ?: return@mapNotNull null
        if (!linked.add(citationKey(url))) return@mapNotNull null
        val label = url.host.removeSurrounding("[", "]")
        "[$label]($destination)"
    }
    return if (links.isEmpty()) "" else links.joinToString(" ", prefix = "\n\n")
}
