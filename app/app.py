"""
EchoChamber - Minimal C7 App
Adds a simple chat tab, an Agent RAG tab, and a Multi-agent thread tab.
Includes an ethics/safety layer: disclaimer + manipulation-risk score + detected factors.
"""

import os
import sys
import json
import html
import re
from pathlib import Path

import gradio as gr
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, AuthenticationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    PROVIDER_PRINCIPAL,
    MODEL_PRINCIPAL,
    PROVIDER_FALLBACK,
    MODEL_FALLBACK,
    TEMPERATURE,
)

from core.agent import generate_agent_response
from core.graph import run_thread


load_dotenv(PROJECT_ROOT / ".env")


BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://openrouter.ai/api/v1",
}


API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "deepseek": os.getenv("OPENROUTER_API_KEY"),
}


ETHICS_DISCLAIMER = """
⚠️ Acesta este un răspuns generat de agenți simulați.
Poate conține bias, opinii, formulări speculative sau limbaj manipulator.
Rezultatele trebuie interpretate ca simulare, nu ca adevăr factual sau recomandare politică.
"""


MANIPULATION_WORDS = [
    "fraudă",
    "frauda",
    "manipulare",
    "manipulați",
    "manipulati",
    "manipulat",
    "trădare",
    "tradare",
    "elite",
    "elitele",
    "control",
    "voturi furate",
    "voturile furate",
    "adevărul va ieși la iveală",
    "adevarul va iesi la iveala",
    "adevărul va ieși",
    "adevarul va iesi",
    "sistemul",
    "minciună",
    "minciuna",
    "propagandă",
    "propaganda",
    "dictatură",
    "dictatura",
    "ocult",
    "globaliști",
    "globalisti",
    "vândut",
    "vandut",
    "corupt",
]


SUSPICIOUS_PATTERNS = [
    "nu mai pot fi manipulați",
    "nu mai pot fi manipulati",
    "adevărul va ieși",
    "adevarul va iesi",
    "adevărul va ieși la iveală",
    "adevarul va iesi la iveala",
    "se fură voturile",
    "se fura voturile",
    "voturi furate",
    "trădare națională",
    "tradare nationala",
    "stat paralel",
    "oculta mondială",
    "oculta mondiala",
    "presa minte",
    "presa manipulează",
    "presa manipuleaza",
]


def clean_agent_text(text):
    """Remove leaked prompt/RAG scaffolding from agent output."""
    if not text:
        return ""

    cleaned = str(text).strip()

    markers = [
        "Simulated response:",
        "Răspuns simulat:",
        "Final response:",
        "Răspuns final:",
    ]

    for marker in markers:
        if marker.lower() in cleaned.lower():
            parts = re.split(re.escape(marker), cleaned, flags=re.IGNORECASE)
            cleaned = parts[-1].strip()

    cleaned = re.sub(r"Agent:\s*[^\n]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Input:\s*.*?(?=Retrieved context:|Context recuperat:|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"Retrieved context:\s*.*?(?=Simulated response:|Răspuns simulat:|Final response:|Răspuns final:|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"Context recuperat:\s*.*?(?=Simulated response:|Răspuns simulat:|Final response:|Răspuns final:|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    cleaned = cleaned.replace("Simulated response:", "")
    cleaned = cleaned.replace("Răspuns simulat:", "")
    cleaned = cleaned.replace("Final response:", "")
    cleaned = cleaned.replace("Răspuns final:", "")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def manipulation_score(text):
    """Calculate a simple manipulation-risk score based on keywords and suspicious patterns."""
    if not text:
        return 0

    text_lower = text.lower()
    score = 0

    for word in MANIPULATION_WORDS:
        if word.lower() in text_lower:
            score += 1

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.lower() in text_lower:
            score += 2

    return score


def risk_label(score):
    """Return a human-readable risk label."""
    if score == 0:
        return "Scăzut"
    elif score <= 2:
        return "Mediu"
    else:
        return "Ridicat"


def make_ethics_report(text):
    """Create a short ethics report for the generated response."""
    text = clean_agent_text(text)

    score = manipulation_score(text)
    label = risk_label(score)

    lower = text.lower()
    factors = []

    if "fraudă" in lower or "frauda" in lower or "voturi furate" in lower:
        factors.append("+ posibilă delegitimare electorală")

    if "sistemul" in lower or "elite" in lower or "elitele" in lower:
        factors.append("+ cadru anti-sistem / anti-elite")

    if "adevărul va ieși" in lower or "adevarul va iesi" in lower:
        factors.append("+ formulare conspiraționistă / speculativă")

    if (
        "manipulat" in lower
        or "manipulați" in lower
        or "manipulati" in lower
        or "manipulare" in lower
    ):
        factors.append("+ limbaj despre manipularea publicului")

    if "trădare" in lower or "tradare" in lower:
        factors.append("+ cadru emoțional de tip trădare / amenințare")

    if "propagandă" in lower or "propaganda" in lower:
        factors.append("+ acuzație de propagandă")

    if not factors:
        factors.append("- nu au fost detectate expresii manipulative evidente")

    factors_text = "\n".join(factors)

    return f"""
{ETHICS_DISCLAIMER}

Scor risc manipulare: {score}
Nivel risc: {label}

Factori detectați:
{factors_text}
"""


def make_client(provider):
    """Create API client for the selected provider."""
    return OpenAI(
        api_key=API_KEYS.get(provider),
        base_url=BASE_URLS.get(provider),
    )


def ask(provider, model, prompt, system=None, temperature=0.7, json_schema=None):
    """Send a prompt to the selected model."""
    client = make_client(provider)

    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    extra_args = {}

    if json_schema:
        extra_args["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra_args,
        )

        text = response.choices[0].message.content.strip()

        if json_schema:
            return json.loads(text)

        return text

    except RateLimitError:
        return f"[Eroare: quota/rate limit pentru modelul {model}.]"

    except AuthenticationError:
        return "[Eroare: API key invalidă sau lipsă. Verifică .env.]"

    except APIError as e:
        return f"[Eroare API: {e}]"

    except Exception as e:
        return f"[Eroare: {type(e).__name__} — {e}]"


def chat(prompt):
    """Simple chat with main provider and fallback."""
    if not prompt.strip():
        return "Scrie un prompt mai întâi."

    answer = ask(
        provider=PROVIDER_PRINCIPAL,
        model=MODEL_PRINCIPAL,
        prompt=prompt,
        temperature=TEMPERATURE,
    )

    if isinstance(answer, str) and answer.startswith("[Eroare"):
        answer = ask(
            provider=PROVIDER_FALLBACK,
            model=MODEL_FALLBACK,
            prompt=prompt,
            temperature=TEMPERATURE,
        )

    return answer


def load_agent_choices():
    """Load available agent slugs from roles.yaml."""
    roles_path = PROJECT_ROOT / "assets" / "roles" / "roles.yaml"

    if not roles_path.exists():
        return []

    with open(roles_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    roles = data["agents"] if "agents" in data else data

    return list(roles.keys())


def rag_agent_response(agent_slug, stimulus, provider, k):
    """Generate an Agent RAG response and return response plus retrieved context and ethics report."""
    if not agent_slug:
        return "Nu există agenți în assets/roles/roles.yaml.", "", ""

    if not stimulus.strip():
        return "Scrie un text politic pentru agent.", "", ""

    try:
        result = generate_agent_response(
            agent_slug=agent_slug,
            stimulus=stimulus,
            provider=provider,
            k=int(k),
            temperature=0.3,
            roles_path="assets/roles/roles.yaml",
        )

        response_text = clean_agent_text(result.get("response", ""))
        rag_text = result.get("rag_text", "")
        ethics_report = make_ethics_report(response_text)

        return response_text, rag_text, ethics_report

    except Exception as e:
        return f"[Eroare Agent RAG: {type(e).__name__} — {e}]", "", ""


def render_thread_html(messages):
    """Render multi-agent messages as styled HTML cards."""
    colors = {
        "anti_suveranist": "#ff7a1a",
        "conspirationist": "#ff4d4d",
        "personalist_salvator": "#b56cff",
        "pro_european": "#4da3ff",
    }

    cards = []

    for msg in messages:
        slug = str(msg.get("slug", ""))
        agent = html.escape(str(msg.get("agent", slug)))
        handle = html.escape(str(msg.get("handle", slug)))
        raw_text = clean_agent_text(msg.get("text", ""))
        text = html.escape(raw_text)
        turn = html.escape(str(msg.get("turn", "")))

        color = colors.get(slug, "#e05a35")

        cards.append(f"""
        <div style="
            border-left:4px solid {color};
            padding:1rem;
            margin:.8rem 0;
            background:#16161a;
            border-radius:10px;
        ">
            <div style="color:{color}; text-transform:uppercase; font-weight:700;">
                {agent}
            </div>
            <div style="font-size:.8rem; color:#999;">
                {handle} · intervenția #{turn}
            </div>
            <p style="color:#ddd; line-height:1.55;">
                {text}
            </p>
        </div>
        """)

    return "\n".join(cards)


def run_multi_agent_thread(
    stimulus,
    provider,
    total_turns,
    use_anti_suveranist,
    use_conspirationist,
    use_personalist_salvator,
    use_pro_european,
):
    """Run the C7 LangGraph multi-agent workflow and return HTML plus ethics report."""
    active_slugs = []

    if use_anti_suveranist:
        active_slugs.append("anti_suveranist")

    if use_conspirationist:
        active_slugs.append("conspirationist")

    if use_personalist_salvator:
        active_slugs.append("personalist_salvator")

    if use_pro_european:
        active_slugs.append("pro_european")

    if not stimulus.strip():
        return "Scrie un text politic mai întâi.", ""

    if not active_slugs:
        return "Selectează cel puțin un agent.", ""

    try:
        messages = run_thread(
            stimulus=stimulus,
            active_slugs=active_slugs,
            total_turns=int(total_turns),
            provider=provider,
            k=3,
        )

        thread_html = render_thread_html(messages)

        all_text = " ".join(
            clean_agent_text(msg.get("text", ""))
            for msg in messages
        )

        ethics_report = make_ethics_report(all_text)

        return thread_html, ethics_report

    except Exception as e:
        return f"[Eroare Multi-agent Thread: {type(e).__name__} — {e}]", ""


agent_choices = load_agent_choices()


with gr.Blocks(title="EchoChamber") as demo:
    gr.Markdown("# EchoChamber")
    gr.Markdown("Aplicație minimă pentru testarea modelelor și a agenților RAG.")

    with gr.Tab("Chat simplu"):
        prompt_box = gr.Textbox(
            label="Prompt",
            value="Explică în 2 propoziții ce este un LLM.",
            lines=4,
        )

        chat_button = gr.Button("Trimite")

        chat_output = gr.Textbox(
            label="Răspuns",
            lines=8,
        )

        chat_button.click(
            fn=chat,
            inputs=prompt_box,
            outputs=chat_output,
        )

    with gr.Tab("Agent RAG"):
        agent_dropdown = gr.Dropdown(
            choices=agent_choices,
            value=agent_choices[0] if agent_choices else None,
            label="Agent",
        )

        provider_dropdown = gr.Dropdown(
            choices=["gemini", "deepseek"],
            value="gemini",
            label="Provider",
        )

        stimulus_box = gr.Textbox(
            label="Text politic nou",
            value="CCR a decis anularea alegerilor după suspiciuni privind influențe externe.",
            lines=4,
        )

        k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Număr fragmente recuperate",
        )

        agent_button = gr.Button("Generează răspuns RAG")

        agent_response_box = gr.Textbox(
            label="Răspuns agent",
            lines=8,
        )

        context_box = gr.Textbox(
            label="Context recuperat",
            lines=12,
        )

        agent_ethics_box = gr.Textbox(
            label="Ethics / Safety Checklist",
            lines=10,
        )

        agent_button.click(
            fn=rag_agent_response,
            inputs=[
                agent_dropdown,
                stimulus_box,
                provider_dropdown,
                k_slider,
            ],
            outputs=[
                agent_response_box,
                context_box,
                agent_ethics_box,
            ],
        )

    with gr.Tab("Multi-agent thread"):
        thread_stimulus = gr.Textbox(
            label="Text politic",
            value="CCR a decis anularea alegerilor după suspiciuni privind influențe externe.",
            lines=4,
        )

        thread_provider = gr.Dropdown(
            choices=["gemini", "deepseek"],
            value="gemini",
            label="Provider",
        )

        thread_turns = gr.Slider(
            minimum=1,
            maximum=4,
            value=1,
            step=1,
            label="Număr intervenții",
        )

        use_anti_suveranist = gr.Checkbox(
            value=True,
            label="Anti-suveranist",
        )

        use_conspirationist = gr.Checkbox(
            value=True,
            label="Conspiraționist",
        )

        use_personalist_salvator = gr.Checkbox(
            value=False,
            label="Personalist salvator",
        )

        use_pro_european = gr.Checkbox(
            value=True,
            label="Pro-european",
        )

        thread_button = gr.Button("Pornește thread")

        thread_output = gr.HTML(label="Thread generat")

        thread_ethics_box = gr.Textbox(
            label="Ethics / Safety Checklist",
            lines=10,
        )

        thread_button.click(
            fn=run_multi_agent_thread,
            inputs=[
                thread_stimulus,
                thread_provider,
                thread_turns,
                use_anti_suveranist,
                use_conspirationist,
                use_personalist_salvator,
                use_pro_european,
            ],
            outputs=[
                thread_output,
                thread_ethics_box,
            ],
        )


if __name__ == "__main__":
    demo.launch()