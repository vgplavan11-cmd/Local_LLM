# challenge3/starter.py
# Memory Agent — Strands SDK + Ollama llama3.2:3b + Mem0 + FAISS
#
# How memory works here:
#   1. Every message you send is added to Mem0's memory store.
#   2. Before every reply, we search Mem0 for memories relevant to your question.
#   3. Those memories are injected into the agent's system prompt so it can recall them.
#   4. FAISS stores the vector index locally on disk — memory persists across runs.

from strands import Agent
from strands.models.ollama import OllamaModel
from mem0 import Memory

# ─────────────────────────────────────────────
# SECTION 1 — Mem0 + FAISS configuration
# ─────────────────────────────────────────────
# We configure three components, all running locally — no API keys needed:
#   • llm       → Ollama llama3.2:3b  (extracts facts worth remembering)
#   • embedder  → Ollama nomic-embed-text  (turns text into vectors)
#   • vector_store → FAISS  (stores those vectors on disk)

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.1,          # low temp → precise fact extraction
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",  # lightweight embedding model
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "memory_agent",
            # Memories are saved here — they survive after you close the terminal.
            "path": "./memory_store",
            # Must match nomic-embed-text's output size exactly (768).
            # Mismatch causes the AssertionError you saw in FAISS.
            "embedding_model_dims": 768,
        },
    },
}

# Initialize the Mem0 memory object using the config above.
# On first run it creates ./memory_store/. On subsequent runs it loads it.
print("Initializing memory store (this may take a moment)...")
memory = Memory.from_config(MEM0_CONFIG)

# A fixed user ID so all memories belong to the same "person".
# Change this if you want to support multiple users.
USER_ID = "user_lavan"

# ─────────────────────────────────────────────
# SECTION 2 — Strands agent (Ollama)
# ─────────────────────────────────────────────
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# We create the base agent without a system prompt here.
# The system prompt is rebuilt dynamically each turn so it includes
# whatever memories are relevant to the current question.
agent = Agent(model=ollama_model)


# ─────────────────────────────────────────────
# SECTION 3 — Helper functions
# ─────────────────────────────────────────────

def store_memory(user_message: str, agent_reply: str) -> None:
    """Save the latest exchange to Mem0.

    Mem0's LLM reads the conversation and decides which facts are worth
    keeping (e.g. names, preferences, dates). Trivial chit-chat is ignored.
    """
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_reply},
    ]
    memory.add(messages, user_id=USER_ID)


def retrieve_memories(query: str, top_k: int = 5) -> list:
    """Search Mem0 for memories semantically related to the query.

    This version of mem0 requires user_id inside filters={} for search(),
    but as a direct kwarg for add(). They differ — don't mix them up.
    """
    results = memory.search(
        query=query,
        filters={"user_id": USER_ID},
        limit=top_k,
    )

    # mem0 returns {"results": [...]} — extract the list.
    if isinstance(results, dict):
        return results.get("results", [])
    return results or []


def build_system_prompt(relevant_memories: list) -> str:
    """Build a system prompt that injects memories so the agent can recall them."""
    base = (
        "You are MemoryBot, a friendly and helpful assistant with a great memory. "
        "You remember things users tell you and use that information naturally in conversation. "
        "Be concise, warm, and personal."
    )
    if not relevant_memories:
        return base

    # Format memories as a bullet list and append to the system prompt.
    mem_lines = "\n".join(f"  - {m['memory']}" for m in relevant_memories)
    return (
        f"{base}\n\n"
        f"What you remember about this user:\n{mem_lines}\n\n"
        "Use these memories to give a personalised, accurate response."
    )


# ─────────────────────────────────────────────
# SECTION 4 — Interactive chat loop
# ─────────────────────────────────────────────
print("\n" + "=" * 52)
print("  MemoryBot — Strands + Ollama + Mem0 + FAISS")
print("=" * 52)
print("Try saying:")
print("  • My name is Lavan")
print("  • I love hiking and coffee")
print("  • What is my name?  (even after restarting!)")
print("  • What do you know about me?")
print("  • Type 'exit' to quit")
print("=" * 52 + "\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! Your memories are saved for next time.")
        break

    if not user_input:
        continue

    # Step 1: Retrieve memories relevant to what the user just said.
    relevant = retrieve_memories(user_input)

    # Step 2: Rebuild the system prompt with those memories injected.
    agent.system_prompt = build_system_prompt(relevant)

    # Step 3: Ask the agent — it now "knows" the memories.
    response = agent(user_input)
    reply_text = str(response)

    # Step 4: Save this exchange so future turns can recall it.
    store_memory(user_input, reply_text)

    print(f"\nMemoryBot: {reply_text}\n")
