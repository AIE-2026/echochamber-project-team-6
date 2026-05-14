# core/agent.py
# ==============
# The Agent class. Each agent:
#   1. Reads its persona from roles.yaml (via config.py)
#   2. Retrieves similar comments from its corpus (RAG via retriever.py)
#   3. Calls the LLM with its system prompt + retrieved context + stimulus
#
# Students: you don't need to modify this file.
# Your work is in assets/roles/roles.yaml — that's where you define the persona.
# Agent module used for RAG-based political community simulation

from core.retriever import Retriever


def generate_agent_response(
    agent_slug,
    stimulus,
    provider="gemini",
    k=5,
    temperature=0.3,
    roles_path="assets/roles/roles.yaml",
):

    # Load retriever
    retriever = Retriever(agent_slug)

    # Retrieve similar fragments
    chunks = retriever.search(stimulus, k=k)

    # Format retrieved context
    rag_text = retriever.format_for_prompt(chunks)

    # Minimal simulated response
    response = f"""
Agent: {agent_slug}

Input:
{stimulus}

Retrieved context:
{rag_text[:1000]}

Simulated response:
Acesta este un răspuns generat pentru agentul {agent_slug}.
"""

    return {
        "response": response,
        "rag_text": rag_text
    }