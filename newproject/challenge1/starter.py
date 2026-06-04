# challenge1/starter.py
# A simple AI agent using Strands SDK + Ollama (llama3.2:3b)

# Import the Agent class from the strands library.
# Agent is the core object that handles conversations with the model.
from strands import Agent

# Import OllamaModel so we can point the agent at our local Ollama server
# instead of a cloud provider like Bedrock or OpenAI.
from strands.models.ollama import OllamaModel

# --- 1. Configure the model ---
# host  : where Ollama is listening (default port is 11434)
# model_id : the exact model tag you pulled with `ollama pull llama3.2:3b`
# temperature : 0.7 gives a nice balance between creativity and accuracy
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# --- 2. Create the agent ---
# We pass our configured model in.
# No extra tools for now — this is the simplest possible agent.
agent = Agent(model=ollama_model)

# --- 3. Give the agent a system prompt ---
# A system prompt sets the agent's personality / role before the conversation starts.
agent.system_prompt = (
    "You are a helpful assistant called StarterBot. "
    "Keep your answers concise and beginner-friendly."
)

# --- 4. Chat loop ---
# We loop so you can keep talking to the agent until you type 'exit'.
print("StarterBot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Skip empty lines
    if not user_input:
        continue

    # Send the message to the agent.
    # The agent returns a response object; printing it shows the text.
    response = agent(user_input)
    print(f"\nStarter Bot: {response}\n")
