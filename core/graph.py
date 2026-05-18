# core/graph.py
# ==============
# LangGraph orchestration for the agentic debate (Tab 3).
#
# HOW THE DEBATE WORKS:
#   This is NOT a fixed round-robin. After each message, a "router" LLM call
#   decides who speaks next based on the thread content. Agents address each
#   other directly by @handle, agree or disagree with previous messages.
#
#   Flow:
#     START → [router] → [agent_X] → [router] → [agent_Y] → ... → END
#
#   The router picks the next agent based on who was just challenged,
#   who hasn't spoken recently, or who has the strongest reaction to give.
#
# Students: you don't need to modify this file.
# core/graph.py
# LangGraph multi-agent workflow for EchoChamber

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from core.agent import generate_agent_response


class ThreadState(TypedDict):
    stimulus: str
    messages: List[Dict[str, Any]]
    active_slugs: List[str]
    total_turns: int
    current_turn: int
    next_slug: str
    provider: str
    k: int


HANDLES = {
    "anti_suveranist": "@EuropaAdevarata",
    "conspirationist": "@HiddenTruthsRO",
    "personalist_salvator": "@VoceaLiderului",
    "pro_european": "@FutureOfEurope",
}


def thread_to_text(messages):
    lines = []

    for message in messages:
        handle = message.get("handle", message.get("slug", "agent"))
        turn = message.get("turn", "?")
        text = message.get("text", "")
        lines.append(f"Turn {turn} — {handle}: {text}")

    return "\n".join(lines)


def pick_next_agent(active_slugs, current_turn):
    if not active_slugs:
        return "__end__"

    index = current_turn % len(active_slugs)
    return active_slugs[index]


def router_node(state: ThreadState):
    if state["current_turn"] >= state["total_turns"]:
        return {"next_slug": "__end__"}

    next_slug = pick_next_agent(
        state["active_slugs"],
        state["current_turn"]
    )

    return {"next_slug": next_slug}


def route_decision(state: ThreadState):
    return state["next_slug"]


def make_agent_node(slug):
    def agent_node(state: ThreadState):
        previous_thread = thread_to_text(state["messages"])

        if previous_thread:
            stimulus_with_context = (
                f"{state['stimulus']}\n\n"
                f"Conversația de până acum:\n{previous_thread}"
            )
        else:
            stimulus_with_context = state["stimulus"]

        result = generate_agent_response(
            agent_slug=slug,
            stimulus=stimulus_with_context,
            provider=state["provider"],
            k=state["k"]
        )

        new_message = {
            "agent": slug,
            "slug": slug,
            "handle": HANDLES.get(slug, slug),
            "text": result["response"],
            "rag_text": result.get("rag_text", ""),
            "turn": state["current_turn"] + 1
        }

        return {
            "messages": state["messages"] + [new_message],
            "current_turn": state["current_turn"] + 1
        }

    return agent_node


def build_graph(active_slugs):
    workflow = StateGraph(ThreadState)

    workflow.add_node("router", router_node)

    for slug in active_slugs:
        workflow.add_node(slug, make_agent_node(slug))

    workflow.add_edge(START, "router")

    route_map = {slug: slug for slug in active_slugs}
    route_map["__end__"] = END

    workflow.add_conditional_edges(
        "router",
        route_decision,
        route_map
    )

    for slug in active_slugs:
        workflow.add_edge(slug, "router")

    return workflow.compile()


def run_thread(
    stimulus,
    active_slugs,
    total_turns=4,
    provider="gemini",
    k=3
):
    graph = build_graph(active_slugs)

    initial_state = {
        "stimulus": stimulus,
        "messages": [],
        "active_slugs": active_slugs,
        "total_turns": total_turns,
        "current_turn": 0,
        "next_slug": "",
        "provider": provider,
        "k": k
    }

    final_state = graph.invoke(initial_state)

    return final_state["messages"]


if __name__ == "__main__":
    messages = run_thread(
        stimulus="CCR a decis anularea alegerilor după suspiciuni privind influențe externe.",
        active_slugs=[
            "anti_suveranist",
            "conspirationist",
            "pro_european"
        ],
        total_turns=1,
        provider="gemini",
        k=1
    )

    print(thread_to_text(messages))