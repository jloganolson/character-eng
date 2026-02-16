import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.chat import ChatSession
from character_eng.prompts import list_characters, load_prompt
from character_eng.world import (
    director_call,
    format_narrator_message,
    load_world_state,
    think_call,
)

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
        world = load_world_state(character)
        system_prompt = load_prompt(character, world_state=world)
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
            elif cmd == "/think":
                handle_think(session, world, label)
                continue
            elif cmd == "/world":
                if world is None:
                    console.print("[dim]No world state for this character.[/dim]")
                else:
                    console.print(world.show())
                continue
            elif cmd.startswith("/world "):
                if world is None:
                    console.print("[dim]No world state for this character.[/dim]")
                    continue
                change_text = user_input[7:]  # preserve original case
                handle_world_change(session, world, change_text, label)
                continue

            # Send to LLM and stream response
            stream_response(session, label, user_input)


def handle_think(session, world, label):
    """Process a /think command — character inner monologue + optional action."""
    console.print("[dim]Thinking...[/dim]")
    try:
        result = think_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
        )
    except Exception as e:
        console.print(f"[red]Think error: {e}[/red]")
        return

    # Display inner monologue
    console.print(f"[dim italic]💭 {result.thought}[/dim italic]")

    if result.action in ("talk", "emote"):
        if result.action == "talk":
            directive = f"[You want to say something unprompted. Directive: {result.action_detail}]"
            nudge = "[The character speaks.]"
        else:
            directive = f"[You feel compelled to act. Directive: {result.action_detail}]"
            nudge = "[The character acts.]"

        session.inject_system(directive)
        stream_response(session, label, nudge)
    else:
        console.print("[dim]No action taken.[/dim]\n")


def handle_world_change(session, world, change_text, label):
    """Process a /world <description> command."""
    console.print(f"[yellow]Director processing:[/yellow] {change_text}")
    try:
        update = director_call(world, change_text)
    except Exception as e:
        console.print(f"[red]Director error: {e}[/red]")
        return

    # Show what changed
    diff_lines = []
    if update.remove_facts:
        for idx in update.remove_facts:
            if 0 <= idx < len(world.dynamic):
                diff_lines.append(f"  [red]- {world.dynamic[idx]}[/red]")
    if update.add_facts:
        for fact in update.add_facts:
            diff_lines.append(f"  [green]+ {fact}[/green]")
    if update.events:
        for event in update.events:
            diff_lines.append(f"  [cyan]~ {event}[/cyan]")
    if diff_lines:
        console.print(Panel("\n".join(diff_lines), title="World Update", border_style="yellow"))

    # Apply the update
    world.apply_update(update)

    # Inject narrator message as system role and trigger character reaction
    narrator_msg = format_narrator_message(update)
    console.print(f"[dim]{narrator_msg}[/dim]")
    session.inject_system(narrator_msg)
    stream_response(session, label, "[React to what just happened.]")


def stream_response(session, label, message):
    """Send a message and stream the response."""
    npc_name = Text(f"{label}: ", style="bold magenta")
    console.print(npc_name, end="")
    for chunk in session.send(message):
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
            "/world          - Show current world state\n"
            "/world <text>   - Describe a change to the world\n"
            "/think          - Character inner monologue + optional action\n"
            "/reload         - Reload prompt files, restart conversation\n"
            "/trace          - Show system prompt, model, token usage\n"
            "/back           - Return to character selection\n"
            "/quit           - Exit program\n"
            "Ctrl+C          - Return to character selection",
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
