"""
EchoChamber - Minimal C6 App
Adds a simple chat tab and an Agent RAG tab.
"""

import os
import sys
import json
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


load_dotenv(PROJECT_ROOT / ".env")

BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
}

API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
}


def make_client(provider):
    """Create API client for the selected provider."""
    return OpenAI(
        api_key=API_KEYS[provider],
        base_url=BASE_URLS[provider],
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
    """Generate an Agent RAG response and return response plus retrieved context."""
    if not agent_slug:
        return "Nu există agenți în assets/roles/roles.yaml.", ""

    if not stimulus.strip():
        return "Scrie un text politic pentru agent.", ""

    try:
        result = generate_agent_response(
            agent_slug=agent_slug,
            stimulus=stimulus,
            provider=provider,
            k=int(k),
            temperature=0.3,
            roles_path="assets/roles/roles.yaml",
        )

        return result["response"], result["rag_text"]

    except Exception as e:
        return f"[Eroare Agent RAG: {type(e).__name__} — {e}]", ""


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

        agent_button.click(
            fn=rag_agent_response,
            inputs=[agent_dropdown, stimulus_box, provider_dropdown, k_slider],
            outputs=[agent_response_box, context_box],
        )


if __name__ == "__main__":
    demo.launch()