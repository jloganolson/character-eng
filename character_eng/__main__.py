import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.chat import ChatSession
from character_eng.prompts import list_characters, load_prompt

console = Console()


def pick_character() -> str | None:
    """Show character menu, return chosen name or None to quit."""
    characters = list_characters()
    if not characters:
        console.print("[red]No characters found in prompts/characters/[/red]")
        return None

    console.print()
    console.print("[bold]Available Characters[/bold]")
    for i, name in enumerate(characters, 1):
        label = name.replace("_", " ").title()
        console.print(f"  [cyan]{i}[/cyan]. {label}")
    console.print(f"  [cyan]q[/cyan]. Quit")
    console.print()

    while True:
        try:
            choice = console.input("[bold]Pick a character:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(characters):
                return characters[idx]
        except ValueError:
            pass
        console.print("[red]Invalid choice, try again.[/red]")


def chat_loop(character: str):
    """Run the chat loop for a character. Returns on /back or Ctrl+C."""
    label = character.replace("_", " ").title()

    # Outer loop: handles /reload by restarting with fresh session
    while True:
        system_prompt = load_prompt(character)
        session = ChatSession(system_prompt)
        console.print(f"\n[bold green]Chatting with {label}[/bold green]  (type /help for commands)\n")

        # Inner loop: conversation turns
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd == "/quit":
                console.print("Goodbye!")
                sys.exit(0)
            elif cmd == "/back":
                return
            elif cmd == "/reload":
                console.print("[yellow]Reloading prompts...[/yellow]")
                break  # break inner loop, outer loop reloads
            elif cmd == "/trace":
                show_trace(session)
                continue
            elif cmd == "/help":
                show_help()
                continue

            # Send to LLM and stream response
            npc_name = Text(f"{label}: ", style="bold magenta")
            console.print(npc_name, end="")
            for chunk in session.send(user_input):
                console.print(chunk, end="", highlight=False)
            console.print("\n")


def show_trace(session: ChatSession):
    """Display trace/debug info in a panel."""
    info = session.trace_info()
    lines = [
        f"[bold]Model:[/bold] {info['model']}",
        f"[bold]History length:[/bold] {info['history_length']} messages",
    ]
    if "total_tokens" in info:
        lines.append(
            f"[bold]Tokens:[/bold] {info['prompt_tokens']} prompt "
            f"+ {info['response_tokens']} response "
            f"= {info['total_tokens']} total"
        )
    lines.append("")
    lines.append("[bold]System Prompt:[/bold]")
    lines.append(info["system_prompt"])
    console.print(Panel("\n".join(lines), title="Trace", border_style="dim"))


def show_help():
    console.print(
        Panel(
            "/reload  - Reload prompt files, restart conversation\n"
            "/trace   - Show system prompt, model, token usage\n"
            "/back    - Return to character selection\n"
            "/quit    - Exit program\n"
            "Ctrl+C   - Return to character selection",
            title="Commands",
            border_style="dim",
        )
    )


def main():
    console.print(Panel("[bold]NPC Character Chat[/bold]", border_style="green"))
    while True:
        character = pick_character()
        if character is None:
            console.print("Goodbye!")
            break
        chat_loop(character)


main()
