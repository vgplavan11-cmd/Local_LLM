# challenge2/starter.py
# Tools Agent — Strands SDK + Ollama llama3.2:3b
# Tools: calculator, weather, age calculator

from datetime import date

# 'Agent' runs the agentic loop; 'tool' turns a plain Python function into
# something the model can discover and call on its own.
from strands import Agent, tool
from strands.models.ollama import OllamaModel


# ─────────────────────────────────────────────
# TOOL 1 — Calculator
# ─────────────────────────────────────────────
# The @tool decorator reads the function name, type hints, and docstring
# to auto-build a JSON schema the model uses to decide when/how to call it.
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression and return the result.

    Use this tool whenever the user asks for any arithmetic calculation
    such as addition, subtraction, multiplication, division, or powers.

    Args:
        expression: A math expression as a string, e.g. '10 * (3 + 4)'

    Returns:
        The numeric result as a string, or an error message.
    """
    try:
        # eval() is safe here because we restrict it to math only —
        # no builtins, no globals.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ─────────────────────────────────────────────
# TOOL 2 — Weather
# ─────────────────────────────────────────────
# In a real project you would call an API (e.g. OpenWeatherMap).
# Here we return mock data so you can focus on how tools wire up.
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this tool when the user asks about the weather or temperature
    in any location.

    Args:
        city: Name of the city, e.g. 'London' or 'New York'

    Returns:
        A short weather summary for that city.
    """
    # Mock weather data — swap this dict for a real API call later.
    mock_data = {
        "london":    "Cloudy, 15°C, humidity 80%",
        "new york":  "Sunny, 22°C, humidity 55%",
        "tokyo":     "Rainy, 18°C, humidity 90%",
        "sydney":    "Clear, 25°C, humidity 60%",
        "paris":     "Partly cloudy, 17°C, humidity 70%",
        "mumbai":    "Hot and humid, 34°C, humidity 85%",
    }
    key = city.lower().strip()
    weather = mock_data.get(key, f"Weather data for '{city}' is not available.")
    return f"Weather in {city}: {weather}"


# ─────────────────────────────────────────────
# TOOL 3 — Age Calculator
# ─────────────────────────────────────────────
@tool
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """Calculate a person's current age from their date of birth.

    Use this tool when the user asks how old someone is, or wants
    to calculate an age from a birth date.

    Args:
        birth_year:  The 4-digit year of birth, e.g. 1995
        birth_month: The month of birth as a number (1-12)
        birth_day:   The day of birth (1-31)

    Returns:
        The person's age in years, or an error message if the date is invalid.
    """
    try:
        dob = date(birth_year, birth_month, birth_day)
        today = date.today()
        # Subtract 1 if the birthday hasn't occurred yet this year.
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
        return f"Age: {age} years old (born {dob.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# ─────────────────────────────────────────────
# AGENT SETUP
# ─────────────────────────────────────────────
# Pass the three tool functions in a list.
# The agent now knows about them and will call them automatically
# whenever the user's question matches a tool's description.
agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],
)

agent.system_prompt = (
    "You are ToolsBot, a helpful assistant with three special abilities:\n"
    "1. You can do math calculations.\n"
    "2. You can look up the weather for a city.\n"
    "3. You can calculate someone's age from their birth date.\n"
    "Always use the right tool for the job. Be concise and friendly."
)

# ─────────────────────────────────────────────
# INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────
print("=" * 50)
print("  ToolsBot — powered by Strands + Ollama")
print("=" * 50)
print("Try asking:")
print("  • What is 25 * 4 + 10?")
print("  • What's the weather in Tokyo?")
print("  • How old is someone born on March 15, 1990?")
print("  • Type 'exit' to quit")
print("=" * 50 + "\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    response = agent(user_input)
    print(f"\nToolsBot: {response}\n")
