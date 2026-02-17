import atexit
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.chat import ChatSession
from character_eng.models import DEFAULT_MODEL, MODELS, PLAN_MODEL
from character_eng.prompts import list_characters, load_prompt
from character_eng.serve import build_vllm_cmd, get_local_models, kill_port
from character_eng.world import (
    Beat,
    EvalResult,
    Goals,
    PlanResult,
    Script,
    director_call,
    eval_call,
    format_narrator_message,
    load_goals,
    load_world_state,
    plan_call,
)

console = Console()

_vllm_process: subprocess.Popen | None = None

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def save_chat_log(character: str, model_config: dict, log: list[dict]):
    """Write chat log to logs/ as timestamped JSON."""
    if not log:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = next((k for k, v in MODELS.items() if v is model_config), "unknown")
    log_path = LOGS_DIR / f"chat_{character}_{model_key}_{timestamp}.json"
    data = {
        "type": "chat",
        "character": character,
        "model": model_key,
        "model_name": model_config["name"],
        "timestamp": datetime.now().isoformat(),
        "entries": log,
    }
    log_path.write_text(json.dumps(data, indent=2))
    console.print(f"[dim]Log: {log_path}[/dim]")


def _stop_vllm():
    """Terminate the managed vLLM process if running."""
    global _vllm_process
    if _vllm_process is not None and _vllm_process.poll() is None:
        console.print("[dim]Stopping vLLM server...[/dim]")
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _vllm_process.kill()
            _vllm_process.wait()
    _vllm_process = None


atexit.register(_stop_vllm)


def _start_local_model() -> dict | None:
    """Show local model sub-menu, start vLLM, return model config or None."""
    global _vllm_process
    local_models = get_local_models()
    if not local_models:
        return None

    # Sub-menu for local model selection
    console.print()
    console.print("[bold]Local Models[/bold]")
    for i, (key, cfg) in enumerate(local_models, 1):
        console.print(f"  [cyan]{i}[/cyan]. {cfg['name']}")
    console.print(f"  [cyan]b[/cyan]. Back")
    console.print()

    while True:
        try:
            choice = console.input("[bold]Pick a local model:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "b":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(local_models):
                break
        except ValueError:
            pass
        console.print("[red]Invalid choice, try again.[/red]")

    key, cfg = local_models[idx]
    path = cfg.get("path", "")
    if not os.path.exists(path):
        console.print(f"[red]Model path not found: {path}[/red]")
        return None

    # Kill existing port, build command, launch
    kill_port("8000")
    cmd = build_vllm_cmd(key, cfg)
    console.print(f"[dim]Starting: {' '.join(cmd)}[/dim]\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        console.print("[red]vllm command not found. Is vLLM installed?[/red]")
        return None

    _vllm_process = proc

    # Stream vLLM output in a background thread
    output_lines: list[str] = []
    def _stream_output():
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            console.print(f"[dim]{line}[/dim]")

    thread = threading.Thread(target=_stream_output, daemon=True)
    thread.start()

    # Poll /health every 2s, with 120s timeout
    health_url = cfg["base_url"].replace("/v1", "/health")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        # Check if process died
        if proc.poll() is not None:
            thread.join(timeout=2)
            console.print("[red]vLLM failed to start.[/red]")
            tail = output_lines[-10:] if output_lines else ["(no output)"]
            for line in tail:
                console.print(f"[dim]  {line}[/dim]")
            _vllm_process = None
            return None

        # Check health
        try:
            urllib.request.urlopen(health_url, timeout=2)
            console.print(f"[green]✓ vLLM ready[/green]")
            return cfg
        except Exception:
            pass

        time.sleep(2)

    # Timeout
    console.print("[red]vLLM not ready after 120s.[/red]")
    tail = output_lines[-10:] if output_lines else ["(no output)"]
    for line in tail:
        console.print(f"[dim]  {line}[/dim]")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    _vllm_process = None
    return None


def _is_local_server_up(base_url: str) -> bool:
    """Quick check if local vLLM server is responding."""
    try:
        health_url = base_url.replace("/v1", "/health")
        urllib.request.urlopen(health_url, timeout=1)
        return True
    except Exception:
        return False


def pick_model() -> dict | None:
    """Show model menu, return chosen config or None to quit.

    Cloud models need their API key env set. Local models need the vLLM server up.
    Auto-selects if only one is available. Offers to start vLLM for unavailable local models.
    """
    available = []
    for key, cfg in MODELS.items():
        if cfg.get("hidden"):
            continue
        if cfg.get("local"):
            if _is_local_server_up(cfg["base_url"]):
                available.append((key, cfg))
        elif os.environ.get(cfg.get("api_key_env", "")):
            available.append((key, cfg))

    # Detect unavailable local models
    local_all = get_local_models()
    local_available_keys = {k for k, _ in available if MODELS[k].get("local")}
    unavailable_local = [(k, c) for k, c in local_all if k not in local_available_keys]
    has_start_option = len(unavailable_local) > 0

    if not available and not has_start_option:
        console.print("[red]No models available. Set an API key in .env or start a local vLLM server.[/red]")
        return None

    if len(available) == 1 and not has_start_option:
        key, cfg = available[0]
        console.print(f"[dim]Using {cfg['name']} (only available model)[/dim]")
        return cfg

    # Show hint about unavailable local models
    if has_start_option:
        n = len(unavailable_local)
        noun = "model" if n == 1 else "models"
        console.print(f"[dim]{n} local {noun} configured but vLLM not running[/dim]")

    console.print()
    console.print("[bold]Available Models[/bold]")
    for i, (key, cfg) in enumerate(available, 1):
        default_tag = " [dim](default)[/dim]" if key == DEFAULT_MODEL else ""
        console.print(f"  [cyan]{i}[/cyan]. {cfg['name']}{default_tag}")
    if has_start_option:
        console.print(f"  [cyan]s[/cyan]. Start local model")
    console.print(f"  [cyan]q[/cyan]. Quit")
    console.print()

    while True:
        try:
            choice = console.input("[bold]Pick a model:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "q":
            return None
        if choice.lower() == "s" and has_start_option:
            result = _start_local_model()
            if result is not None:
                return result
            # Fall back to menu on failure — re-display
            return pick_model()
        if choice == "":
            # Default selection
            for key, cfg in available:
                if key == DEFAULT_MODEL:
                    return cfg
            if available:
                return available[0][1]
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx][1]
        except ValueError:
            pass
        console.print("[red]Invalid choice, try again.[/red]")


def pick_character() -> str | None:
    """Show character menu, return chosen name or None to quit.

    Auto-selects if only one character exists.
    """
    characters = list_characters()
    if not characters:
        console.print("[red]No characters found in prompts/characters/[/red]")
        return None
    if len(characters) == 1:
        label = characters[0].replace("_", " ").title()
        console.print(f"[dim]Using {label} (only available character)[/dim]")
        return characters[0]

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


def get_plan_model_config():
    """Look up the plan model config, return None if API key not set."""
    cfg = MODELS.get(PLAN_MODEL)
    if cfg is None:
        return None
    api_key_env = cfg.get("api_key_env", "")
    if api_key_env and not os.environ.get(api_key_env):
        return None
    return cfg


def chat_loop(character: str, model_config: dict):
    """Run the chat loop for a character. Returns on /back or Ctrl+C."""
    label = character.replace("_", " ").title()
    log: list[dict] = []
    plan_model_config = get_plan_model_config()

    if plan_model_config:
        plan_label = MODELS[PLAN_MODEL]["name"]
        console.print(f"[dim]Planner: {plan_label}[/dim]")
    else:
        console.print(f"[dim]Planner: unavailable (no {MODELS.get(PLAN_MODEL, {}).get('api_key_env', 'API key')} set)[/dim]")

    # Outer loop: handles /reload by restarting with fresh session
    while True:
        world = load_world_state(character)
        goals = load_goals(character)
        system_prompt = load_prompt(character, world_state=world)
        session = ChatSession(system_prompt, model_config)
        script = Script()

        # Generate initial script at boot
        plan_result = run_plan(session, world, goals, "", plan_model_config)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)

        console.print(f"\n[bold green]Chatting with {label} ({model_config['name']})[/bold green]  (type /help for commands)\n")

        # Inner loop: conversation turns
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                save_chat_log(character, model_config, log)
                return

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd == "/quit":
                save_chat_log(character, model_config, log)
                console.print("Goodbye!")
                sys.exit(0)
            elif cmd == "/back":
                save_chat_log(character, model_config, log)
                return
            elif cmd == "/reload":
                console.print("[yellow]Reloading prompts...[/yellow]")
                log.append({"type": "reload"})
                break  # break inner loop, outer loop reloads
            elif cmd == "/trace":
                show_trace(session)
                continue
            elif cmd == "/help":
                show_help()
                continue
            elif cmd == "/think":
                handle_think(session, world, goals, script, label, model_config, plan_model_config, log)
                continue
            elif cmd == "/plan":
                console.print(script.show())
                continue
            elif cmd == "/goals":
                if goals is None:
                    console.print("[dim]No goals for this character.[/dim]")
                else:
                    console.print(goals.show())
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
                handle_world_change(session, world, goals, script, change_text, label, model_config, plan_model_config, log)
                continue

            # --- Turn flow ---

            # Bootstrap: if no script, call planner to generate one
            if script.is_empty():
                plan_result = run_plan(session, world, goals, "", plan_model_config)
                if plan_result and plan_result.beats:
                    apply_plan(script, plan_result)

            if script.current_beat is not None:
                # Normal path: inject beat, chat, then eval
                inject_beat(session, script.current_beat)

                response = stream_response(session, label, user_input)
                entry = {"type": "send", "input": user_input, "response": response}

                eval_result = run_eval(session, world, goals, script, model_config)
                if eval_result:
                    display_eval(eval_result)
                    inject_eval(session, eval_result)
                    entry["eval"] = eval_to_dict(eval_result)

                    if eval_result.script_status == "advance":
                        script.advance()
                        if script.current_beat:
                            console.print(f"[magenta]  next beat:  [{script.current_beat.intent}][/magenta]")
                        else:
                            console.print("[dim]  script complete, replanning...[/dim]")
                            plan_result = run_plan(session, world, goals, "", plan_model_config)
                            if plan_result and plan_result.beats:
                                apply_plan(script, plan_result)
                    elif eval_result.script_status == "off_book" and eval_result.plan_request:
                        plan_result = run_plan(session, world, goals, eval_result.plan_request, plan_model_config)
                        if plan_result and plan_result.beats:
                            apply_plan(script, plan_result)

                log.append(entry)
            else:
                # No script available (planner unavailable/failed) — just chat + eval
                response = stream_response(session, label, user_input)
                entry = {"type": "send", "input": user_input, "response": response}

                eval_result = run_eval(session, world, goals, script, model_config)
                if eval_result:
                    display_eval(eval_result)
                    inject_eval(session, eval_result)
                    entry["eval"] = eval_to_dict(eval_result)

                log.append(entry)


def run_eval(session, world, goals, script, model_config):
    """Run eval_call, return EvalResult or None on error."""
    console.print("[dim]Evaluating...[/dim]")
    try:
        return eval_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            model_config=model_config,
            goals=goals,
            script=script,
        )
    except Exception as e:
        console.print(f"[red]Eval error: {e}[/red]")
        return None


def run_plan(session, world, goals, plan_request, plan_model_config):
    """Run plan_call, return PlanResult or None on error."""
    if plan_model_config is None:
        console.print("[dim]Planner unavailable, skipping.[/dim]")
        return None
    console.print("[dim]Planning...[/dim]")
    try:
        return plan_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            goals=goals,
            plan_request=plan_request,
            plan_model_config=plan_model_config,
        )
    except Exception as e:
        console.print(f"[red]Plan error: {e}[/red]")
        return None


def display_eval(result):
    """Display all eval fields."""
    console.print(f"[dim]  thought:    {result.thought}[/dim]")
    console.print(f"[dim]  gaze:       {result.gaze}[/dim]")
    console.print(f"[dim]  expression: {result.expression}[/dim]")
    console.print(f"[dim]  status:     {result.script_status}[/dim]")
    if result.plan_request:
        console.print(f"[dim]  plan_req:   {result.plan_request}[/dim]")


def eval_to_dict(result):
    """Convert EvalResult to dict for logging."""
    d = {
        "thought": result.thought,
        "gaze": result.gaze,
        "expression": result.expression,
        "script_status": result.script_status,
    }
    if result.bootstrap_line:
        d["bootstrap_line"] = result.bootstrap_line
        d["bootstrap_intent"] = result.bootstrap_intent
    if result.plan_request:
        d["plan_request"] = result.plan_request
    return d


def inject_eval(session, result):
    """Inject eval result into session history so future evals can build on it."""
    parts = [f"[Inner thought: {result.thought}]"]
    if result.gaze:
        parts.append(f"[gaze:{result.gaze}]")
    if result.expression:
        parts.append(f"[emote:{result.expression}]")
    session.inject_system(" ".join(parts))


def inject_beat(session, beat):
    """Inject beat directive before chat."""
    directive = f'[Your current objective: {beat.intent}. You want to say: "{beat.line}". Adapt naturally.]'
    session.inject_system(directive)


def apply_plan(script, plan_result):
    """Replace script beats, display summary."""
    script.replace(plan_result.beats)
    console.print(f"[magenta]Script loaded ({len(plan_result.beats)} beats):[/magenta]")
    for i, beat in enumerate(plan_result.beats):
        marker = "→" if i == 0 else " "
        console.print(f"[magenta]  {marker} {i}. [{beat.intent}] \"{beat.line}\"[/magenta]")


def handle_think(session, world, goals, script, label, model_config, plan_model_config, log):
    """Process a /think command — eval + optional beat delivery + optional planner."""
    result = run_eval(session, world, goals, script, model_config)
    if not result:
        return

    display_eval(result)
    inject_eval(session, result)

    entry = {"type": "eval", **eval_to_dict(result)}

    if script.current_beat and result.script_status == "advance":
        # Advance and deliver next beat
        script.advance()
        if script.current_beat:
            console.print(f"[magenta]  next beat:  [{script.current_beat.intent}][/magenta]")
            inject_beat(session, script.current_beat)
            response = stream_response(session, label, "[The character speaks.]")
            entry["response"] = response
        else:
            console.print("[dim]  script complete, replanning...[/dim]")
            plan_result = run_plan(session, world, goals, "", plan_model_config)
            if plan_result and plan_result.beats:
                apply_plan(script, plan_result)
    elif script.current_beat and result.script_status == "hold":
        # Deliver current beat
        inject_beat(session, script.current_beat)
        response = stream_response(session, label, "[The character speaks.]")
        entry["response"] = response
    elif result.script_status == "off_book" and result.plan_request:
        # Replan
        plan_result = run_plan(session, world, goals, result.plan_request, plan_model_config)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)
    else:
        console.print("[dim]No action taken.[/dim]\n")

    log.append(entry)


def handle_world_change(session, world, goals, script, change_text, label, model_config, plan_model_config, log):
    """Process a /world <description> command."""
    console.print(f"[yellow]Director processing:[/yellow] {change_text}")
    try:
        update = director_call(world, change_text, model_config)
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

    # Inject narrator message as system role
    narrator_msg = format_narrator_message(update)
    console.print(f"[dim]{narrator_msg}[/dim]")
    session.inject_system(narrator_msg)

    # Eval before reacting
    eval_result = run_eval(session, world, goals, script, model_config)
    if eval_result:
        display_eval(eval_result)
        inject_eval(session, eval_result)

    # Inject beat if available
    if script.current_beat:
        inject_beat(session, script.current_beat)

    # Character reacts
    response = stream_response(session, label, "[React to what just happened.]")

    entry = {
        "type": "world",
        "input": change_text,
        "update": {
            "remove_facts": update.remove_facts,
            "add_facts": update.add_facts,
            "events": update.events,
        },
        "narrator": narrator_msg,
        "response": response,
    }
    if eval_result:
        entry["eval"] = eval_to_dict(eval_result)
        # Check if world change triggers replan
        if eval_result.script_status == "off_book" and eval_result.plan_request:
            plan_result = run_plan(session, world, goals, eval_result.plan_request, plan_model_config)
            if plan_result and plan_result.beats:
                apply_plan(script, plan_result)
    log.append(entry)


def stream_response(session, label, message) -> str:
    """Send a message and stream the response. Returns full response text."""
    npc_name = Text(f"{label}: ", style="bold magenta")
    console.print(npc_name, end="")
    full_response = []
    for chunk in session.send(message):
        full_response.append(chunk)
        console.print(chunk, end="", highlight=False)
    console.print("\n")
    return "".join(full_response)


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
            "/think          - Character eval + optional beat delivery\n"
            "/plan           - Show current script (beat list)\n"
            "/goals          - Show character goals\n"
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
    try:
        model_config = pick_model()
        if model_config is None:
            console.print("Goodbye!")
            return
        while True:
            character = pick_character()
            if character is None:
                console.print("Goodbye!")
                break
            chat_loop(character, model_config)
    finally:
        _stop_vllm()


if __name__ == "__main__":
    main()
