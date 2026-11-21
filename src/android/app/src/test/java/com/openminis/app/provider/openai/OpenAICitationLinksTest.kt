package com.openminis.app.provider.openai

import org.junit.Assert.assertEquals
import org.junit.Test

class OpenAICitationLinksTest {
    private val docs = "https://developers.openai.com/api/docs/guides/tools-web-search?api-mode=responses"

    @Test fun alreadyLinkedSourceStaysInline() {
        val answer = "官方文档。([developers.openai.com]($docs))"
        assertEquals("", openAICitationFooter(answer, listOf("$docs&utm_source=openai", docs)))
    }

    @Test fun missingSourcesUseShortLabelsAndKeepDistinctDestinations() {
        val other = "https://developers.openai.com/guide_(tools)?a=1#intro"
        val answer = "[文档]($docs)"
        assertEquals("\n\n[developers.openai.com](https://developers.openai.com/guide_%28tools%29?a=1#intro)",
            openAICitationFooter(answer, listOf(docs, other, other)))
        assertEquals("\n\n[developers.openai.com]($docs)", openAICitationFooter("developers.openai.com", listOf(docs)))
    }
}
