# core/config.py
# ===============
# Central configuration: loads roles.yaml and .env API keys.
# All other modules import from here instead of reading files directly.
#
# Students: you don't need to modify this file.
# If you want to add a new LLM provider, add it to AVAILABLE_MODELS below.

# Model principal
PROVIDER_PRINCIPAL = "gemini"
MODEL_PRINCIPAL = "gemini-2.5-flash-lite"

# Model fallback
PROVIDER_FALLBACK = "openrouter"
MODEL_FALLBACK = "openrouter/free"

# Temperature
TEMPERATURE = 0.1