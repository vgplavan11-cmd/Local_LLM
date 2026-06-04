# challenge4/starter.py
# Full Agent — Strands SDK + Ollama llama3.2:3b + Tools + Mem0 + FAISS
#
# This agent combines everything from challenge2 and challenge3:
#   • 3 tools    → calculator, weather, age calculator  (challenge2)
#   • Memory     → Mem0 + FAISS, fully local            (challenge3)
#
# Flow per turn:
#   User input
#     → retrieve relevant memories from FAISS
#     → inject memories into system prompt
#     → agent decides which tool(s) to call (if any)
#     → reply is shown + saved back to Mem0

from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from mem0 import Memory


# ═══════════════════════════════════════════════════════
# SECTION 1 — Tools
# ═══════════════════════════════════════════════════════
# @tool turns a plain Python function into something the model
# can discover and call on its own, based on its docstring + type hints.

# ── Tool 1: Calculator ──────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression and return the result.

    Use this tool whenever the user asks for any arithmetic:
    addition, subtraction, multiplication, division, or powers.

    Args:
        expression: A math expression string, e.g. '10 * (3 + 4)'

    Returns:
        The numeric result, or an error message.
    """
    try:
        # eval() is sandboxed — no builtins means no import tricks.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ── Tool 2: Weather ─────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this tool when the user asks about weather or temperature
    in any location.

    Args:
        city: Name of the city, e.g. 'London' or 'Tokyo'

    Returns:
        A short weather summary for that city.
    """
    # Mock data — replace with a real API call (e.g. OpenWeatherMap) later.
    mock_data = {
        "london":    "Cloudy, 15°C, humidity 80%",
        "new york":  "Sunny, 22°C, humidity 55%",
        "tokyo":     "Rainy, 18°C, humidity 90%",
        "sydney":    "Clear, 25°C, humidity 60%",
        "paris":     "Partly cloudy, 17°C, humidity 70%",
        "mumbai":    "Hot and humid, 34°C, humidity 85%",
        "chennai":   "Sunny, 36°C, humidity 75%",
        "delhi":     "Hazy, 38°C, humidity 50%",
    }
    weather = mock_data.get(city.lower().strip(),
                            f"Weather data for '{city}' is not available.")
    return f"Weather in {city}: {weather}"


# ── Tool 3: Age Calculator ──────────────────────────────
@tool
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """Calculate a person's current age from their date of birth.

    Use this tool when the user asks how old someone is or wants
    to calculate an age from a birth date.

    Args:
        birth_year:  4-digit year, e.g. 1995
        birth_month: Month as a number (1-12)
        birth_day:   Day of the month (1-31)

    Returns:
        The person's age in years, or an error if the date is invalid.
    """
    try:
        dob = date(birth_year, birth_month, birth_day)
        today = date.today()
        # Subtract 1 if the birthday hasn't happened yet this calendar year.
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
        return f"Age: {age} years old (born {dob.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# ═══════════════════════════════════════════════════════
# SECTION 2 — Memory configuration (Mem0 + FAISS)
# ═══════════════════════════════════════════════════════
# All three components run locally — no API keys needed:
#   llm          → extracts facts worth remembering from the conversation
#   embedder     → converts those facts into 768-dim vectors
#   vector_store → FAISS saves those vectors to disk in ./memory_store/

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.1,           # low = precise fact extraction
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",   # 768-dim embedding model
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "full_agent",
            "path": "./memory_store",      # persists across runs
            "embedding_model_dims": 768,   # MUST match nomic-embed-text output
        },
    },
}

print("Initializing memory store...")
# First run → creates ./memory_store/
# Subsequent runs → loads existing index from disk
memory = Memory.from_config(MEM0_CONFIG)

# All memories are tagged to this user ID.
USER_ID = "user_lavan"


# ═══════════════════════════════════════════════════════
# SECTION 3 — Memory helper functions
# ═══════════════════════════════════════════════════════

def store_memory(user_message: str, agent_reply: str) -> None:
    """Pass the conversation turn to Mem0.

    Mem0's LLM reads it and decides what facts are worth keeping.
    Trivial messages like 'ok' are ignored automatically.

    NOTE: add() takes user_id as a direct kwarg.
    """
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_reply},
    ]
    memory.add(messages, user_id=USER_ID)


def retrieve_memories(query: str, top_k: int = 5) -> list:
    """Semantic search — find the top_k memories closest to the query.

    NOTE: search() requires user_id inside filters={}, NOT as a direct kwarg.
    This is intentionally different from add() — don't mix them up.
    """
    results = memory.search(
        query=query,
        filters={"user_id": USER_ID},
        limit=top_k,
    )
    # mem0 returns {"results": [...]} — unwrap it.
    if isinstance(results, dict):
        return results.get("results", [])
    return results or []


def build_system_prompt(relevant_memories: list) -> str:
    """Build the system prompt, injecting any relevant memories.

    Without memories → plain system prompt.
    With memories    → same prompt + bullet list of what we know about the user.
    The agent uses this context to give personalised, tool-aware replies.
    """
    base = (
        "You are FullAgent, a smart and friendly assistant with:\n"
        "  1. A calculator — for any math questions.\n"
        "  2. A weather tool — for weather in any city.\n"
        "  3. An age calculator — to compute age from a birth date.\n"
        "  4. A memory — you remember things users tell you.\n\n"
        "Always use the right tool when needed. "
        "Use your memories to give personalised responses. "
        "Be concise and warm."
    )

    if not relevant_memories:
        return base

    mem_lines = "\n".join(f"  - {m['memory']}" for m in relevant_memories)
    return (
        f"{base}\n\n"
        f"What you remember about this user:\n{mem_lines}\n\n"
        "Use these memories naturally in your response."
    )


# ═══════════════════════════════════════════════════════
# SECTION 4 — Agent setup
# ═══════════════════════════════════════════════════════
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# Pass all three tools in the list.
# The system prompt is set dynamically each turn (see chat loop below),
# so we don't set it here.
agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],
)


# ═══════════════════════════════════════════════════════
# SECTION 5 — Interactive chat loop
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 54)
print("  FullAgent — Strands + Ollama + Tools + Memory")
print("=" * 54)
print("Try asking:")
print("  • My name is Lavan")
print("  • What is 128 / 4 + 7?")
print("  • What's the weather in Mumbai?")
print("  • How old is someone born on July 4, 1995?")
print("  • What is my name?  ← memory recall")
print("  • What do you know about me?")
print("  • Type 'exit' to quit")
print("=" * 54 + "\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! Your memories are saved for next time.")
        break

    if not user_input:
        continue

    # Step 1 — Retrieve memories relevant to this message.
    relevant = retrieve_memories(user_input)

    # Step 2 — Rebuild system prompt with those memories injected.
    agent.system_prompt = build_system_prompt(relevant)

    # Step 3 — Agent replies (calls a tool automatically if needed).
    response = agent(user_input)
    reply_text = str(response)

    # Step 4 — Save this exchange so future turns can recall it.
    store_memory(user_input, reply_text)

    print(f"\nFullAgent: {reply_text}\n")
