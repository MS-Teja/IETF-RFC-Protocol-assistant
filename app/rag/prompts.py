"""Prompt templates for the RAG pipeline.

All prompts live here — never inlined in pipeline.py or routes.
This directly satisfies assessment Section 8: "Modify prompts" should
be a single-file change.
"""

SYSTEM_PROMPT = """You are a Senior Protocol Engineer and IETF RFC specialist. You answer questions about web and API protocols (HTTP/1.1–HTTP/3, TLS 1.3, OAuth 2.0, JWT, WebSockets) using ONLY the provided RFC excerpts.

RULES:
1. Ground every claim in the provided context. Do NOT use outside knowledge.
2. If the context is insufficient, say so explicitly — never fabricate.
3. Cite sources using the bracket labels provided, e.g. "[RFC 9110, §15.5.1]".
4. When multiple RFCs apply, compare them (e.g. HTTP/2 vs HTTP/3 framing).
5. Flag any "implementation-defined" or "MAY" language — don't present optional behaviour as mandatory.
6. Be precise but readable. Prefer concrete examples over abstract descriptions.

FORMATTING:
- Use Markdown: headings, bullet lists, `inline code` for protocol tokens/headers/methods.
- Use fenced code blocks for wire-format examples or header sequences.
- Keep answers focused. Lead with the direct answer, then supporting detail."""


USER_PROMPT_TEMPLATE = """RETRIEVED RFC CONTEXT:
{context}

---

QUESTION: {query}

Answer using ONLY the context above. Cite specific RFC sections with bracket notation."""


CONVERSATION_CONTEXT_TEMPLATE = """The following is the recent conversation for context. Use it to understand follow-up questions, but ground all factual claims in the RETRIEVED RFC CONTEXT only.

CONVERSATION:
{conversation}

---

"""


def build_context_string(chunks: list[dict]) -> str:
    """Build the context string with label injection for each chunk.

    Each chunk enters the prompt as "[RFC 9110, §15.5.1 Bad Request] <text>"
    so citations come from retrieval metadata, not model memory.
    """
    if not chunks:
        return "No relevant context was retrieved."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.get("metadata", {}).get("label", f"[Source {i}]")
        content = chunk.get("content", "")
        score = chunk.get("score", 0.0)
        context_parts.append(f"{label} (relevance: {score:.2f})\n{content}")

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """Build the full user prompt with context, conversation history, and query."""
    context = build_context_string(chunks)

    conversation_block = ""
    if history:
        # Include last 4 messages (2 exchanges) for LLM context
        recent = history[-4:] if len(history) >= 4 else history
        conv_parts = []
        for msg in recent:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Truncate long assistant responses to save tokens
            if role == "Assistant" and len(content) > 400:
                content = content[:400] + "..."
            conv_parts.append(f"{role}: {content}")
        conversation_text = "\n".join(conv_parts)
        conversation_block = CONVERSATION_CONTEXT_TEMPLATE.format(conversation=conversation_text)

    return conversation_block + USER_PROMPT_TEMPLATE.format(context=context, query=query)
