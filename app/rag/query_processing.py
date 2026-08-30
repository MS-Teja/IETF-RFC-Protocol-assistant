"""Query processing — contextualize follow-up queries.

Rewrites follow-up messages into standalone queries before embedding.
A question like "what about withdrawing it?" embeds with almost no signal
on its own — the pronoun carries meaning the embedding model can't resolve.
"""

from app.utils.logging import logger


def contextualize_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up query into a standalone query using conversation history.

    Uses a cheap heuristic: prepends the last 1-2 exchanges to give the
    embedding model enough context to resolve pronouns and references.

    Upgrade path: swap this for a small LLM call doing real query rewriting.

    Args:
        query: The current user query.
        history: List of message dicts with 'role' and 'content' keys.

    Returns:
        The standalone query string for embedding.
    """
    if not history:
        return query

    # Take the last 2 messages (1 exchange) for context
    recent = history[-2:] if len(history) >= 2 else history

    # Build context string from recent history
    context_parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate long messages to keep embedding query reasonable
        if len(content) > 200:
            content = content[:200] + "..."
        context_parts.append(f"{role}: {content}")

    context = " | ".join(context_parts)
    rewritten = f"{context} | Current question: {query}"

    logger.debug("Query contextualized: '%s' → '%s'", query, rewritten[:100])
    return rewritten
