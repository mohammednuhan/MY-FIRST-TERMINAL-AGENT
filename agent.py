import os
from dotenv import load_dotenv
from openai import OpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align
console = Console()

# Load .env
load_dotenv()

# Conversation history
history = []

# Connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


#Welcome message
console.print(
    Align.center(
        """
        ╔══════════════════════════════════════════╗
        ║                                          ║
        ║             N U H A N                    ║
        ║                                          ║
        ║        ◉  A G E N T   C O R E  ◉        ║
        ║                                          ║
        ║        [ SYSTEM INITIALIZED ]            ║
        ║                                          ║
        ║        INPUT  ──────►  THINK             ║
        ║                         │                ║
        ║                         ▼                ║
        ║                      EXECUTE             ║
        ║                         │                ║
        ║                         ▼                ║
        ║                      OUTPUT              ║
        ║                                          ║
        ╚══════════════════════════════════════════╝
        """,
        style="bold bright_cyan"
    )
)

console.print(
    Align.center("[bold white]TERMINAL INTELLIGENCE[/bold white]")
)

console.print(
    Align.center("[dim]v0.1 • local interface • online[/dim]")
)

console.print()


# Terminal loop
while True:

    # Get user input
    user_input = console.input("[bold green]nuhan> [/bold green] ")

    if not user_input.strip():
        continue

    # Exit
    if user_input == "/exit":
        break

    # Add user message to history
    history.append({
        "role": "user",
        "content": user_input
    })

    # Send conversation to AI
    response = client.chat.completions.create(
        model="poolside/laguna-s-2.1:free",
        messages=history
    )

    # Get AI answer
    answer = response.choices[0].message.content

    # Display AI answer
    console.print("[bold cyan]Agent:[/bold cyan]")
    console.print(Markdown(answer))

    # Add AI answer to history
    history.append({
        "role": "assistant",
        "content": answer
    })