import argparse
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.chat import ChatSession
from character_eng.config import load_config
from character_eng.models import BIG_MODEL, CHAT_MODEL, MODELS
from character_eng.perception import PerceptionEvent, load_sim_script, process_perception
from character_eng.person import PeopleState
from character_eng.prompts import list_characters, load_prompt
from character_eng.qa_personas import _action_badge, _esc, annotation_assets
from character_eng.scenario import DirectorResult, load_scenario_script, director_call
from character_eng.world import (
    EvalResult,
    Goals,
    PlanResult,
    Script,
    WorldUpdate,
    condition_check_call,
    eval_call,
    expression_call,
    format_pending_narrator,
    load_beat_guide,
    load_goals,
    load_world_state,
    plan_call,
    reconcile_call,
)

console = Console()

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# --- Background reconciliation threading ---
_reconcile_lock = threading.Lock()
_reconcile_result: WorldUpdate | None = None
_reconcile_thread: threading.Thread | None = None

# --- Background eval threading ---
_eval_lock = threading.Lock()
_eval_result: EvalResult | None = None
_eval_thread: threading.Thread | None = None
_eval_version: int = 0

# --- Background plan threading ---
_plan_lock = threading.Lock()
_plan_result: PlanResult | None = None
_plan_thread: threading.Thread | None = None
_plan_version: int = 0

# --- Background director threading ---
_director_lock = threading.Lock()
_director_result: DirectorResult | None = None
_director_thread: threading.Thread | None = None
_director_version: int = 0

# --- Context version counter (main thread only) ---
_context_version: int = 0


def _bump_version():
    """Increment context version. Called on user messages, /world changes, /beat delivery."""
    global _context_version
    _context_version += 1


def _run_reconcile(world: "WorldState", pending: list[str], model_config: dict, people=None):
    """Background thread target. Calls reconcile_call, stores result under lock."""
    global _reconcile_result
    try:
        result = reconcile_call(world, pending, model_config, people=people)
        with _reconcile_lock:
            _reconcile_result = result
    except Exception as e:
        console.print(f"[red]Background reconcile error: {e}[/red]")


def _start_reconcile(world, model_config, big_model_config, people=None):
    """Drain pending changes and spawn background reconcile thread."""
    global _reconcile_thread
    pending = world.clear_pending()
    if not pending:
        return
    # Use plan model if available, fall back to chat model
    cfg = big_model_config if big_model_config else model_config
    _reconcile_thread = threading.Thread(
        target=_run_reconcile, args=(world, pending, cfg, people), daemon=True
    )
    _reconcile_thread.start()


def _check_reconcile(world, log, people=None):
    """Check if background reconcile result is ready. If so, apply and display."""
    global _reconcile_result, _reconcile_thread
    with _reconcile_lock:
        result = _reconcile_result
        _reconcile_result = None

    if result is None:
        return

    _reconcile_thread = None

    # Show diff panel
    diff_lines = []
    if result.remove_facts:
        for fid in result.remove_facts:
            text = world.dynamic.get(fid, fid)
            diff_lines.append(f"  [red]- {fid}. {text}[/red]")
    if result.add_facts:
        for fact in result.add_facts:
            diff_lines.append(f"  [green]+ {fact}[/green]")
    if result.events:
        for event in result.events:
            diff_lines.append(f"  [cyan]~ {event}[/cyan]")
    if result.person_updates:
        for pu in result.person_updates:
            for fid in pu.remove_facts:
                diff_lines.append(f"  [red]- {pu.person_id}: {fid}[/red]")
            for fact in pu.add_facts:
                diff_lines.append(f"  [green]+ {pu.person_id}: {fact}[/green]")
            if pu.set_name:
                diff_lines.append(f"  [blue]  {pu.person_id} name → {pu.set_name}[/blue]")
            if pu.set_presence:
                diff_lines.append(f"  [blue]  {pu.person_id} presence → {pu.set_presence}[/blue]")
    if diff_lines:
        console.print(Panel("\n".join(diff_lines), title="Reconciled", border_style="yellow"))

    world.apply_update(result)
    if people is not None and result.person_updates:
        people.apply_updates(result.person_updates)

    log.append({
        "type": "reconcile",
        "remove_facts": result.remove_facts,
        "add_facts": result.add_facts,
        "events": result.events,
    })


# --- Background eval functions ---

def _run_eval(system_prompt, world, history, model_config, goals, script, people=None, stage_goal=""):
    """Background thread target for eval."""
    global _eval_result
    try:
        result = eval_call(
            system_prompt=system_prompt,
            world=world,
            history=history,
            model_config=model_config,
            goals=goals,
            script=script,
            people=people,
            stage_goal=stage_goal,
        )
        with _eval_lock:
            _eval_result = result
    except Exception as e:
        console.print(f"[red]Background eval error: {e}[/red]")


def _start_eval(session, world, goals, script, model_config, people=None, stage_goal=""):
    """Kick off background eval. Skips if one is already in flight."""
    global _eval_thread, _eval_version
    if _eval_thread is not None and _eval_thread.is_alive():
        return
    _eval_version = _context_version
    _eval_thread = threading.Thread(
        target=_run_eval,
        args=(session.system_prompt, world, session.get_history(), model_config, goals, script, people, stage_goal),
        daemon=True,
    )
    _eval_thread.start()


def _check_eval(session, script, log):
    """Check if background eval is ready. Apply if not stale.
    Returns (needs_plan, plan_request) tuple."""
    global _eval_result, _eval_thread
    with _eval_lock:
        result = _eval_result
        _eval_result = None

    if result is None:
        return (False, "")

    _eval_thread = None

    if _eval_version != _context_version:
        console.print("[dim](eval result discarded — stale)[/dim]")
        return (False, "")

    display_eval(result)
    inject_eval(session, result)
    log.append({"type": "eval", **eval_to_dict(result)})

    if result.script_status == "advance":
        script.advance()
        if script.current_beat:
            console.print(f"[magenta]  next beat:  [{script.current_beat.intent}][/magenta]")
            return (False, "")
        else:
            console.print("[dim]  script complete, replanning...[/dim]")
            return (True, "")
    elif result.script_status == "off_book" and result.plan_request:
        return (True, result.plan_request)

    return (False, "")


# --- Background plan functions ---

def _run_plan_bg(system_prompt, world, history, goals, plan_request, big_model_config, people=None, stage_goal=""):
    """Background thread target for plan."""
    global _plan_result
    try:
        result = plan_call(
            system_prompt=system_prompt,
            world=world,
            history=history,
            goals=goals,
            plan_request=plan_request,
            plan_model_config=big_model_config,
            people=people,
            stage_goal=stage_goal,
        )
        with _plan_lock:
            _plan_result = result
    except Exception as e:
        console.print(f"[red]Background plan error: {e}[/red]")


def _start_plan(session, world, goals, plan_request, big_model_config, people=None, stage_goal=""):
    """Kick off background plan. Skips if planner unavailable or one already in flight."""
    global _plan_thread, _plan_version
    if big_model_config is None:
        return
    if _plan_thread is not None and _plan_thread.is_alive():
        return
    _plan_version = _context_version
    _plan_thread = threading.Thread(
        target=_run_plan_bg,
        args=(session.system_prompt, world, session.get_history(), goals, plan_request, big_model_config, people, stage_goal),
        daemon=True,
    )
    _plan_thread.start()


def _check_plan(script):
    """Check if background plan is ready. Apply if not stale."""
    global _plan_result, _plan_thread
    with _plan_lock:
        result = _plan_result
        _plan_result = None

    if result is None:
        return

    _plan_thread = None

    if _plan_version != _context_version:
        console.print("[dim](plan result discarded — stale)[/dim]")
        return

    if result.beats:
        apply_plan(script, result)


# --- Background director functions ---

def _run_director(scenario, world, people, history, model_config):
    """Background thread target for director."""
    global _director_result
    try:
        result = director_call(scenario, world, people, history, model_config)
        with _director_lock:
            _director_result = result
    except Exception as e:
        console.print(f"[red]Background director error: {e}[/red]")


def _start_director(scenario, world, people, session, model_config):
    """Kick off background director. Skips if one is already in flight or no scenario."""
    global _director_thread, _director_version
    if scenario is None:
        return
    if _director_thread is not None and _director_thread.is_alive():
        return
    _director_version = _context_version
    _director_thread = threading.Thread(
        target=_run_director,
        args=(scenario, world, people, session.get_history(), model_config),
        daemon=True,
    )
    _director_thread.start()


def _check_director(scenario, script, session, world, goals, people, big_model_config):
    """Check if background director result is ready. Returns whether a stage transition occurred."""
    global _director_result, _director_thread
    with _director_lock:
        result = _director_result
        _director_result = None

    if result is None:
        return False

    _director_thread = None

    if _director_version != _context_version:
        console.print("[dim](director result discarded — stale)[/dim]")
        return False

    if result.status == "advance" and result.exit_index >= 0 and scenario is not None:
        stage = scenario.active_stage
        if stage and 0 <= result.exit_index < len(stage.exits):
            exit_obj = stage.exits[result.exit_index]
            if scenario.advance_to(exit_obj.goto):
                new_stage = scenario.active_stage
                console.print(f"[cyan]  director: {result.thought}[/cyan]")
                if new_stage:
                    console.print(f"[bold cyan]Stage → {new_stage.name}: {new_stage.goal}[/bold cyan]")
                    # Trigger replanning with new stage goal
                    stage_goal = new_stage.goal if new_stage else ""
                    _start_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)
                return True
    else:
        if result.thought:
            console.print(f"[dim]  director: {result.thought}[/dim]")

    return False


def save_chat_log(character: str, model_config: dict, log: list[dict], session_id: str = ""):
    """Write chat log to logs/ as timestamped JSON."""
    if not log:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = next((k for k, v in MODELS.items() if v is model_config), "unknown")
    log_path = LOGS_DIR / f"chat_{character}_{model_key}_{session_id}_{timestamp}.json"
    data = {
        "type": "chat",
        "character": character,
        "model": model_key,
        "model_name": model_config["name"],
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "entries": log,
    }
    log_path.write_text(json.dumps(data, indent=2))
    console.print(f"[dim]Log: {log_path}[/dim]")


def save_chat_html(character: str, model_config: dict, log: list[dict], session_id: str = ""):
    """Write chat log as an annotatable HTML report to logs/."""
    if not log:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "annotated").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = next((k for k, v in MODELS.items() if v is model_config), "unknown")
    report_name = f"chat_{character}_{model_key}_{session_id}_{timestamp}"
    html_path = LOGS_DIR / f"{report_name}.html"
    label = character.replace("_", " ").title()

    # Map log entries to turn divs
    turns_html = ""
    turn_num = 0
    pending_extras: list[dict] = []  # eval/reconcile entries to attach to next turn

    for entry in log:
        etype = entry.get("type", "")

        # Accumulate eval/reconcile/plan entries as extras for the preceding turn
        if etype in ("eval", "reconcile", "plan", "eval_stale", "plan_stale", "reload"):
            pending_extras.append(entry)
            continue

        if etype not in ("send", "world", "beat", "see"):
            continue

        action = etype
        if etype == "send":
            action = "message"
        input_text = entry.get("input", "")
        response = entry.get("response", "")

        # Input display
        if action == "beat":
            input_html = '<div class="beat-input">/beat</div>'
        elif action == "world":
            input_html = f'<div class="world-input">/world {_esc(input_text)}</div>'
        elif action == "see":
            input_html = f'<div class="world-input">/see {_esc(input_text)}</div>'
        else:
            input_html = f'<div class="user-input">{_esc(input_text)}</div>'

        # Response
        resp_html = ""
        if response:
            resp_html = f'<div class="npc-response">{_esc(response)}</div>'

        # Expression metadata (gaze/expression from post-processing)
        expr_gaze = entry.get("gaze", "")
        expr_gaze_type = entry.get("gaze_type", "hold")
        expr_expression = entry.get("expression", "")
        if expr_gaze or expr_expression:
            gaze_display = _esc(expr_gaze)
            if expr_gaze_type == "glance":
                gaze_display += " (glance)"
            resp_html += (
                f'<div class="eval-details" style="margin-top:0.2em;font-size:0.85em;color:#8b949e;">'
                f'gaze: {gaze_display} &middot; expression: {_esc(expr_expression)}'
                f'</div>'
            )

        # Extras from preceding eval/reconcile entries
        extras_html = ""
        for ex in pending_extras:
            ex_type = ex.get("type", "")
            if ex_type == "eval":
                extras_html += (
                    f'<details class="eval-details"><summary>eval: {_esc(ex.get("script_status", "?"))}</summary>'
                    f'<div class="eval-body">'
                    f'<div><b>thought:</b> {_esc(ex.get("thought", ""))}</div>'
                    f'<div><b>status:</b> {_esc(ex.get("script_status", ""))}</div>'
                )
                if ex.get("plan_request"):
                    extras_html += f'<div><b>plan_request:</b> {_esc(ex["plan_request"])}</div>'
                extras_html += "</div></details>"
            elif ex_type == "reconcile":
                removals = ", ".join(ex.get("remove_facts", []))
                additions = ", ".join(ex.get("add_facts", []))
                events = ", ".join(ex.get("events", []))
                extras_html += (
                    f'<details class="eval-details"><summary>reconcile</summary>'
                    f'<div class="eval-body">'
                    f'<div><b>removed:</b> {_esc(removals)}</div>'
                    f'<div><b>added:</b> {_esc(additions)}</div>'
                    f'<div><b>events:</b> {_esc(events)}</div>'
                    f'</div></details>'
                )
            elif ex_type == "eval_stale":
                extras_html += ' <span class="stale">STALE DISCARD (eval)</span>'
            elif ex_type == "plan_stale":
                extras_html += ' <span class="stale">STALE DISCARD (plan)</span>'
        pending_extras.clear()

        ann_key = f"{label}:{turn_num}"
        ann_key_safe = ann_key.replace(" ", "_").replace(":", "_")
        turns_html += (
            f'<div class="turn" data-persona="{_esc(label)}" data-turn="{turn_num}"'
            f' data-action="{_esc(action)}" data-input="{_esc(input_text)}"'
            f' data-response="{_esc(response)}">'
            f'<div class="turn-header">Turn {turn_num + 1} {_action_badge(action)}</div>'
            f'{input_html}{resp_html}{extras_html}'
            f'<button class="annotate-btn" data-ann-key="{_esc(ann_key)}"'
            f' onclick="toggleAnnotation(\'{_esc(label)}\',{turn_num})">&#9998; annotate</button>'
            f'<div class="annotation-area" id="ann-{ann_key_safe}">'
            f'<textarea placeholder="Add a note..."'
            f' oninput="updateAnnotation(\'{_esc(label)}\',{turn_num},this.value)"></textarea>'
            f'</div>'
            f'</div>'
        )
        turn_num += 1

    ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Chat — {_esc(label)} — {_esc(model_key)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, system-ui, 'Segoe UI', sans-serif;
    max-width: 900px; margin: 2em auto; padding: 0 1em;
    background: #0d1117; color: #c9d1d9;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 0.3em; }}
  .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 1.5em; }}
  .conversation {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 0.5em 1em;
  }}
  .turn {{ padding: 0.6em 0; border-bottom: 1px solid #21262d; }}
  .turn:last-child {{ border-bottom: none; }}
  .turn-header {{ font-size: 0.85em; color: #8b949e; margin-bottom: 0.4em; }}
  .badge {{
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 0.8em; color: #fff; font-weight: 600;
  }}
  .stale {{
    background: #d29922; color: #0d1117; padding: 1px 6px;
    border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-left: 0.5em;
  }}
  .user-input {{ color: #58a6ff; margin: 0.3em 0; padding-left: 1em; }}
  .world-input {{ color: #d29922; margin: 0.3em 0; padding-left: 1em; font-style: italic; }}
  .beat-input {{ color: #bc8cff; margin: 0.3em 0; padding-left: 1em; font-style: italic; }}
  .npc-response {{
    color: #7ee787; margin: 0.3em 0; padding-left: 1em;
    border-left: 3px solid #238636; padding: 0.3em 0.8em;
    background: #0d1117; border-radius: 0 4px 4px 0;
  }}
  .eval-details {{ margin: 0.3em 0 0 1em; }}
  .eval-details summary {{ cursor: pointer; color: #8b949e; font-size: 0.85em; }}
  .eval-body {{
    padding: 0.5em; background: #0d1117; border-radius: 4px;
    font-size: 0.85em; margin-top: 0.3em;
  }}
  .eval-body div {{ margin: 0.2em 0; }}
</style></head><body>
<h1>Chat Session — {_esc(label)}</h1>
<p class="meta">{ts_display} &middot; model: <b>{_esc(model_config['name'])}</b> (<code>{_esc(model_key)}</code>) &middot; session: <code>{session_id}</code></p>
<div class="conversation">{turns_html}</div>
{annotation_assets(report_name, model_key, character)}
</body></html>"""

    html_path.write_text(html)
    console.print(f"[dim]HTML: {html_path}[/dim]")



def _voice_dev_kw(voice_cfg):
    """Build kwargs dict from voice config for device display.

    Resolves string device names to indices via resolve_device().
    """
    kw = {}
    if voice_cfg is not None:
        if voice_cfg.input_device is not None:
            from character_eng.voice import resolve_device
            kw["input_device"] = resolve_device(voice_cfg.input_device, "input")
        if voice_cfg.output_device is not None:
            from character_eng.voice import resolve_device
            kw["output_device"] = resolve_device(voice_cfg.output_device, "output")
    return kw


def _create_voice_io(voice_cfg, VoiceIO_cls):
    """Create a VoiceIO instance from voice config."""
    kw = _voice_dev_kw(voice_cfg)
    if voice_cfg is not None:
        if not voice_cfg.aec:
            kw["aec"] = False
        kw["tts_backend"] = voice_cfg.tts_backend
        kw["ref_audio"] = voice_cfg.ref_audio
        kw["ref_text"] = voice_cfg.ref_text
        kw["tts_model"] = voice_cfg.tts_model
        kw["tts_device"] = voice_cfg.tts_device
        kw["tts_server_url"] = voice_cfg.tts_server_url
        kw["pocket_voice"] = voice_cfg.pocket_voice
    return VoiceIO_cls(**kw)


def _show_voice_devices(dev_kw, get_default_devices, list_audio_devices):
    """Print the active mic/speaker devices after voice init."""
    devs = list_audio_devices() if dev_kw else None

    if "input_device" in dev_kw:
        dev = next((d for d in devs if d["index"] == dev_kw["input_device"]), None)
        if dev:
            console.print(f"[dim]  Mic:     [{dev['index']}] {dev['name']}[/dim]")
    else:
        input_dev, _ = get_default_devices()
        if input_dev:
            console.print(f"[dim]  Mic:     [{input_dev['index']}] {input_dev['name']}[/dim]")

    if "output_device" in dev_kw:
        if devs is None:
            devs = list_audio_devices()
        dev = next((d for d in devs if d["index"] == dev_kw["output_device"]), None)
        if dev:
            console.print(f"[dim]  Speaker: [{dev['index']}] {dev['name']}[/dim]")
    else:
        _, output_dev = get_default_devices()
        if output_dev:
            console.print(f"[dim]  Speaker: [{output_dev['index']}] {output_dev['name']}[/dim]")


def show_stage_hud(scenario, voice_active: bool = False):
    """Print compact stage HUD line: [stage] 1: label  2: label ... (+ hotkey legend in voice mode)"""
    if scenario is None or scenario.active_stage is None:
        if voice_active:
            console.print("  [dim]i:info  b:beat  t:trace  Esc:text  q:quit[/dim]")
        return
    stage = scenario.active_stage
    parts = []
    for i, exit in enumerate(stage.exits):
        lbl = exit.label or exit.condition[:20]
        parts.append(f"{i + 1}: {lbl}")
    exits_str = "  ".join(parts)
    if voice_active:
        exits_str += "  [dim]i:info  b:beat  t:trace  Esc:text  q:quit[/dim]"
    console.print(f"  [cyan]\\[{stage.name}][/cyan] {exits_str}")


def handle_trigger(trigger_num, scenario, session, world, goals, script, people, label, model_config, big_model_config, eval_model_config, log, voice_io=None):
    """Handle a number-key trigger (1-4) to advance the scenario via the chosen exit."""
    if scenario is None or scenario.active_stage is None:
        console.print("[dim]No scenario loaded.[/dim]")
        return
    stage = scenario.active_stage
    idx = trigger_num - 1
    if idx < 0 or idx >= len(stage.exits):
        console.print(f"[red]No exit {trigger_num} (stage has {len(stage.exits)} exits)[/red]")
        return

    exit_obj = stage.exits[idx]
    old_stage = stage.name

    # 1. Advance scenario
    if not scenario.advance_to(exit_obj.goto):
        console.print(f"[red]Cannot advance to {exit_obj.goto}[/red]")
        return

    new_stage = scenario.active_stage
    console.print(f"[bold cyan]Stage {old_stage} → {new_stage.name}[/bold cyan]")

    # 2. Inject exit condition as perception event → character reacts
    handle_perception(
        session, world, goals, script, people, scenario,
        exit_obj.condition, label,
        model_config, big_model_config, eval_model_config, log,
        voice_io=voice_io,
    )

    # 3. Clear stale script and synchronously replan with new stage goal
    #    Must be synchronous — if async, /beat would deliver stale beats from the old stage.
    if voice_io is not None:
        voice_io._cancel_auto_beat()
    script.replace([])
    stage_goal = new_stage.goal if new_stage else ""
    plan_result = run_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)
    if plan_result and plan_result.beats:
        apply_plan(script, plan_result)


def show_info(world, script, goals, scenario, people):
    """Show all state: world, plan, goals, stage, people."""
    if world is not None:
        console.print(world.show())
    else:
        console.print("[dim]No world state.[/dim]")
    console.print(script.show())
    if goals is not None:
        console.print(goals.show())
    if scenario is not None:
        console.print(scenario.show())
    console.print(people.show())


def get_chat_model_config() -> dict | None:
    """Look up the chat model config, return None if API key not set."""
    cfg = MODELS.get(CHAT_MODEL)
    if cfg is None:
        return None
    api_key_env = cfg.get("api_key_env", "")
    if api_key_env and not os.environ.get(api_key_env):
        return None
    return cfg


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


def get_big_model_config():
    """Look up the big model config, return None if API key not set."""
    cfg = MODELS.get(BIG_MODEL)
    if cfg is None:
        return None
    api_key_env = cfg.get("api_key_env", "")
    if api_key_env and not os.environ.get(api_key_env):
        return None
    return cfg


def chat_loop(character: str, model_config: dict, voice_mode: bool = False, voice_cfg=None):
    """Run the chat loop for a character. Returns on /back or Ctrl+C."""
    label = character.replace("_", " ").title()
    session_id = uuid.uuid4().hex[:8]
    log: list[dict] = []
    big_model_config = get_big_model_config()

    # Big model handles eval, plan, and reconcile; falls back to chat model
    eval_model_config = big_model_config if big_model_config else model_config

    if big_model_config:
        big_label = MODELS[BIG_MODEL]["name"]
        console.print(f"[dim]Big model: {big_label} (eval + plan + reconcile)[/dim]")
    else:
        console.print(f"[dim]Big model: unavailable (no {MODELS.get(BIG_MODEL, {}).get('api_key_env', 'API key')} set), using chat model[/dim]")

    # --- Voice setup ---
    voice_io = None
    if voice_mode:
        from character_eng.voice import VoiceIO, check_voice_available, get_default_devices, list_audio_devices, VOICE_OFF, EXIT, VOICE_ERROR
        available, reason = check_voice_available(tts_backend=voice_cfg.tts_backend if voice_cfg else "elevenlabs")
        if available:
            voice_io = _create_voice_io(voice_cfg, VoiceIO)
            try:
                voice_io.start()
                _backend = voice_cfg.tts_backend if voice_cfg else "elevenlabs"
                _tts_labels = {"local": "Local Qwen3-TTS", "qwen": "Local Qwen3-TTS", "pocket": "Pocket-TTS", "elevenlabs": "ElevenLabs"}
                _tts_label = _tts_labels.get(_backend, _backend)
                console.print(f"[green]Voice mode active[/green] — speak to chat, Escape to toggle, hotkeys: i/b/t/1-4")
                console.print(f"[dim]TTS: {_tts_label}[/dim]")
                _show_voice_devices(_voice_dev_kw(voice_cfg), get_default_devices, list_audio_devices)
            except Exception as e:
                console.print(f"[red]Voice init failed: {e}[/red]")
                console.print("[dim]Falling back to text mode[/dim]")
                voice_io = None
        else:
            console.print(f"[red]Voice unavailable: {reason}[/red]")
            console.print("[dim]Falling back to text mode[/dim]")

    # Outer loop: handles /reload by restarting with fresh session
    while True:
        world = load_world_state(character)
        goals = load_goals(character)
        people = PeopleState()
        scenario = load_scenario_script(character)
        system_prompt = load_prompt(character, world_state=world, people_state=people)
        session = ChatSession(system_prompt, model_config)
        script = Script()

        if scenario:
            console.print(f"[dim]Scenario: {scenario.name} (stage: {scenario.current_stage})[/dim]")

        # Get stage goal for planner context
        stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""

        # Generate initial script at boot (synchronous)
        plan_result = run_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)

        had_user_input = False
        _arm_auto_beat = True  # arm on startup so first beat fires without user input

        mode_tag = " [green]voice[/green]" if voice_io is not None else ""
        console.print(f"\n[bold green]Chatting with {label} ({model_config['name']})[/bold green]{mode_tag}  [dim]{session_id}[/dim]  (type /help for commands)\n")

        # Inner loop: conversation turns
        while True:
            # --- Get input (voice or text) ---
            show_stage_hud(scenario, voice_active=(voice_io is not None))
            if voice_io is not None and _arm_auto_beat:
                # Centralized auto-beat: start timer here so it fires after all
                # console output is done and TTS echo has settled (avoids Deepgram
                # false positives cancelling the timer mid-countdown).
                voice_io._start_auto_beat()
                _arm_auto_beat = False
                try:
                    user_input = voice_io.wait_for_input()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    voice_io.stop()
                    voice_io = None
                    save_chat_log(character, model_config, log, session_id)
                    save_chat_html(character, model_config, log, session_id)
                    return

                # User/hotkey input arrived — cancel auto-beat timer and drain
                # any stale /beat that fired during the previous command.
                # Let actual /beat through so auto-beat chain works.
                if user_input != "/beat":
                    voice_io.cancel_and_drain_auto_beat()

                # Handle sentinel strings from voice
                from character_eng.voice import VOICE_OFF, EXIT, VOICE_ERROR
                if user_input == EXIT:
                    console.print()
                    if voice_io is not None:
                        voice_io.stop()
                        voice_io = None
                    save_chat_log(character, model_config, log, session_id)
                    save_chat_html(character, model_config, log, session_id)
                    console.print("Goodbye!")
                    sys.exit(0)
                elif user_input == VOICE_OFF:
                    console.print("\n[yellow]Voice off — text mode[/yellow]")
                    voice_io.stop()
                    voice_io = None
                    continue
                elif user_input == VOICE_ERROR:
                    console.print("\n[red]Voice error — switching to text mode[/red]")
                    voice_io.stop()
                    voice_io = None
                    continue

                # Show what was transcribed or which hotkey was pressed
                if user_input.startswith("/"):
                    console.print(f"[dim]  → {user_input}[/dim]")
                elif user_input in ("1", "2", "3", "4"):
                    console.print(f"[dim]  → trigger {user_input}[/dim]")
                else:
                    console.print(f"[bold blue]You[/bold blue] [dim]{session_id}[/dim]: {user_input}")
            else:
                try:
                    user_input = console.input(f"[bold blue]You[/bold blue] [dim]{session_id}[/dim]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    if voice_io is not None:
                        voice_io.stop()
                        voice_io = None
                    save_chat_log(character, model_config, log, session_id)
                    save_chat_html(character, model_config, log, session_id)
                    return

            if not user_input:
                continue

            # --- Turn start: check background results ---
            stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""

            if world is not None:
                _check_reconcile(world, log, people=people)

            needs_plan, plan_request = _check_eval(session, script, log)
            if needs_plan:
                _start_plan(session, world, goals, plan_request, big_model_config, people=people, stage_goal=stage_goal)

            _check_director(scenario, script, session, world, goals, people, big_model_config)

            _check_plan(script)

            # --- Command dispatch ---
            cmd = user_input.lower()
            if cmd == "/quit":
                if voice_io is not None:
                    voice_io.stop()
                    voice_io = None
                save_chat_log(character, model_config, log, session_id)
                save_chat_html(character, model_config, log, session_id)
                console.print("Goodbye!")
                sys.exit(0)
            elif cmd == "/back":
                if voice_io is not None:
                    voice_io.stop()
                    voice_io = None
                save_chat_log(character, model_config, log, session_id)
                save_chat_html(character, model_config, log, session_id)
                return
            elif cmd == "/reload":
                console.print("[yellow]Reloading prompts...[/yellow]")
                log.append({"type": "reload"})
                break  # break inner loop, outer loop reloads
            elif cmd == "/trace":
                show_trace(session)
                continue
            elif cmd == "/help":
                show_help(voice_io is not None)
                continue
            elif cmd == "/devices":
                try:
                    from character_eng.voice import list_audio_devices
                    devices = list_audio_devices()
                    lines = []
                    for d in devices:
                        tags = []
                        if d["is_default_input"]:
                            tags.append("default input")
                        if d["is_default_output"]:
                            tags.append("default output")
                        tag_str = f"  [green]({', '.join(tags)})[/green]" if tags else ""
                        io = f"{d['inputs']}in/{d['outputs']}out"
                        lines.append(f"  [cyan]{d['index']:>2}[/cyan]. {d['name']}  [dim]({io})[/dim]{tag_str}")
                    console.print(Panel("\n".join(lines), title="Audio Devices", border_style="dim"))
                except ImportError:
                    console.print("[red]Voice packages not installed. Run: uv sync --extra voice[/red]")
                continue
            elif cmd == "/voice":
                if voice_io is not None:
                    console.print("[yellow]Voice off — text mode[/yellow]")
                    voice_io.stop()
                    voice_io = None
                else:
                    from character_eng.voice import VoiceIO, check_voice_available, get_default_devices, list_audio_devices
                    available, reason = check_voice_available(tts_backend=voice_cfg.tts_backend if voice_cfg else "elevenlabs")
                    if available:
                        voice_io = _create_voice_io(voice_cfg, VoiceIO)
                        try:
                            voice_io.start()
                            _backend = voice_cfg.tts_backend if voice_cfg else "elevenlabs"
                            _tts_labels = {"local": "Local Qwen3-TTS", "qwen": "Local Qwen3-TTS", "pocket": "Pocket-TTS", "elevenlabs": "ElevenLabs"}
                            _tts_label = _tts_labels.get(_backend, _backend)
                            console.print("[green]Voice mode active[/green] — speak to chat, Escape to toggle, hotkeys: i/b/t/1-4")
                            console.print(f"[dim]TTS: {_tts_label}[/dim]")
                            _show_voice_devices(_voice_dev_kw(voice_cfg), get_default_devices, list_audio_devices)
                        except Exception as e:
                            console.print(f"[red]Voice init failed: {e}[/red]")
                            voice_io = None
                    else:
                        console.print(f"[red]Voice unavailable: {reason}[/red]")
                continue
            elif cmd == "/beat":
                handle_beat(session, world, goals, script, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, people=people, scenario=scenario)
                _arm_auto_beat = True
                continue
            elif cmd == "/info":
                show_info(world, script, goals, scenario, people)
                continue
            elif cmd.startswith("/world "):
                if world is None:
                    console.print("[dim]No world state for this character.[/dim]")
                    continue
                change_text = user_input[7:]  # preserve original case
                handle_world_change(session, world, goals, script, change_text, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, people=people, scenario=scenario)
                _arm_auto_beat = True
                continue
            elif cmd.startswith("/see "):
                if world is None:
                    console.print("[dim]No world state for this character.[/dim]")
                    continue
                see_text = user_input[5:]  # preserve original case
                handle_perception(session, world, goals, script, people, scenario, see_text, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io)
                _arm_auto_beat = True
                continue
            elif cmd.startswith("/sim "):
                sim_name = user_input[5:].strip()
                if not sim_name:
                    console.print("[red]Usage: /sim <name>[/red]")
                    continue
                run_sim(sim_name, character, session, world, goals, script, people, scenario, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io)
                _arm_auto_beat = True
                continue
            elif cmd in ("1", "2", "3", "4"):
                handle_trigger(int(cmd), scenario, session, world, goals, script, people, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io)
                _arm_auto_beat = True
                continue
            elif user_input.startswith("/"):
                console.print(f"[red]Unknown command: {user_input.split()[0]}[/red]")
                console.print("[dim]Type /help to see available commands[/dim]")
                continue

            # --- Barge-in context injection ---
            if voice_io is not None and voice_io._barged_in:
                session.inject_system(
                    "[The user interrupted you mid-sentence. They want your attention. "
                    "Acknowledge the interruption briefly, then listen. Keep it very short.]"
                )

            # --- Turn flow ---

            # First user input: natural LLM response, then background eval + replan
            if not had_user_input:
                had_user_input = True
                response = stream_response(session, label, user_input, voice_io=voice_io, expr_model_config=model_config)
                expr = stream_response._last_expr
                entry = {"type": "send", "input": user_input, "response": response}
                if expr:
                    entry["gaze"] = expr.gaze
                    entry["gaze_type"] = expr.gaze_type
                    entry["expression"] = expr.expression
                log.append(entry)
                _bump_version()
                _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
                _start_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)
                _start_director(scenario, world, people, session, eval_model_config)
                _arm_auto_beat = True
                continue

            # Bootstrap: if no script, start background plan and fall through to unguided
            if script.is_empty():
                _start_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)

            if script.current_beat is not None:
                # Normal path: LLM-guided beat delivery — reacts to user while serving intent
                response = stream_guided_beat(session, script.current_beat, label, user_input, voice_io=voice_io, expr_model_config=model_config)
            else:
                # No script available (planner unavailable/failed) — LLM generates response
                response = stream_response(session, label, user_input, voice_io=voice_io, expr_model_config=model_config)
            expr = stream_response._last_expr
            entry = {"type": "send", "input": user_input, "response": response}
            if expr:
                entry["gaze"] = expr.gaze
                entry["gaze_type"] = expr.gaze_type
                entry["expression"] = expr.expression
            log.append(entry)
            _bump_version()
            _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
            _start_director(scenario, world, people, session, eval_model_config)
            _arm_auto_beat = True


def run_eval(session, world, goals, script, model_config):
    """Run eval_call synchronously, return EvalResult or None on error."""
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


def run_plan(session, world, goals, plan_request, big_model_config, people=None, stage_goal=""):
    """Run plan_call synchronously, return PlanResult or None on error."""
    if big_model_config is None:
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
            plan_model_config=big_model_config,
            people=people,
            stage_goal=stage_goal,
        )
    except Exception as e:
        console.print(f"[red]Plan error: {e}[/red]")
        return None


def display_eval(result):
    """Display all eval fields."""
    console.print(f"[dim]  thought:    {result.thought}[/dim]")
    console.print(f"[dim]  status:     {result.script_status}[/dim]")
    if result.plan_request:
        console.print(f"[dim]  plan_req:   {result.plan_request}[/dim]")


def eval_to_dict(result):
    """Convert EvalResult to dict for logging."""
    d = {
        "thought": result.thought,
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
    session.inject_system(f"[Inner thought: {result.thought}]")


def run_expression(session, model_config, line=None):
    """Run expression_call on a character line. Display + inject + return for logging.

    If line is None, scans session history for the last assistant message.
    """
    if line is None:
        history = session.get_history()
        for msg in reversed(history):
            if msg["role"] == "assistant":
                line = msg["content"]
                break
    if not line:
        return None

    try:
        result = expression_call(line, model_config)
    except Exception as e:
        console.print(f"[dim]  expression error: {e}[/dim]")
        return None

    if result.gaze or result.expression:
        gaze_label = result.gaze
        if result.gaze_type == "glance":
            gaze_label += " (glance)"
        console.print(f"  gaze: {gaze_label}  expression: {result.expression}")

    parts = []
    if result.gaze:
        if result.gaze_type == "glance":
            parts.append(f"[glance:{result.gaze}]")
        else:
            parts.append(f"[gaze:{result.gaze}]")
    if result.expression:
        parts.append(f"[emote:{result.expression}]")
    if parts:
        session.inject_system(" ".join(parts))

    return result


def deliver_beat(session, beat, label, voice_io=None):
    """Deliver a pre-rendered beat as the character's response. Returns the beat line."""
    npc_name = Text(f"{label}: ", style="bold magenta")
    console.print(npc_name, end="")
    console.print(beat.line)
    console.print()
    if voice_io is not None:
        voice_io.speak_text(beat.line)
        if voice_io._barged_in:
            ratio = voice_io.speaker_playback_ratio
            spoken_chars = max(int(len(beat.line) * ratio), 1)
            if spoken_chars < len(beat.line):
                truncated = beat.line[:spoken_chars] + " \u2014"
                session.add_assistant(truncated)
                return truncated
    session.add_assistant(beat.line)
    return beat.line


def stream_guided_beat(session, beat, label, user_input, voice_io=None, expr_model_config=None) -> str:
    """Use the beat's intent to guide an LLM response that reacts to user input.

    Injects beat guidance as a system message, then streams a real LLM response.
    The LLM sees the intent and example line, but generates its own dialogue that
    acknowledges what the user said. Returns the full response text.
    """
    # Inject beat guidance so the LLM knows the intent
    guidance = load_beat_guide(beat.intent, beat.line)
    session.inject_system(guidance)

    # Stream a real LLM response guided by the beat intent
    return stream_response(session, label, user_input, voice_io=voice_io, expr_model_config=expr_model_config)


def apply_plan(script, plan_result):
    """Replace script beats, display summary."""
    script.replace(plan_result.beats)
    console.print(f"[magenta]Script loaded ({len(plan_result.beats)} beats):[/magenta]")
    for i, beat in enumerate(plan_result.beats):
        marker = "→" if i == 0 else " "
        console.print(f"[magenta]  {marker} {i}. [{beat.intent}] \"{beat.line}\"[/magenta]")


def handle_beat(session, world, goals, script, label, model_config, big_model_config, eval_model_config, log, voice_io=None, people=None, scenario=None):
    """Process a /beat command — condition-gated beat delivery (time passes)."""
    entry = {"type": "beat"}

    # 1. If no current beat, replan (synchronous — user explicitly asked for a beat)
    if script.current_beat is None:
        plan_result = run_plan(session, world, goals, "", big_model_config)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)
        if script.current_beat is None:
            console.print("[dim]No beats available.[/dim]\n")
            log.append(entry)
            return

    beat = script.current_beat

    # 2. If beat has a condition, check it (synchronous — gates delivery)
    if beat.condition:
        console.print(f"[dim]Checking condition: {beat.condition}[/dim]")
        try:
            cond_result = condition_check_call(
                condition=beat.condition,
                system_prompt=session.system_prompt,
                world=world,
                history=session.get_history(),
                model_config=model_config,
            )
        except Exception as e:
            console.print(f"[red]Condition check error: {e}[/red]")
            log.append(entry)
            return

        entry["condition"] = beat.condition
        entry["condition_met"] = cond_result.met

        if not cond_result.met:
            # Display idle line as character response
            if cond_result.idle:
                npc_name = f"{label}: "
                console.print(f"[bold magenta]{npc_name}[/bold magenta]{cond_result.idle}")
                console.print()
                idle_text = cond_result.idle
                if voice_io is not None:
                    voice_io.speak_text(idle_text)
                    if voice_io._barged_in:
                        ratio = voice_io.speaker_playback_ratio
                        spoken = max(int(len(idle_text) * ratio), 1)
                        if spoken < len(idle_text):
                            idle_text = idle_text[:spoken] + " \u2014"
                session.add_assistant(idle_text)
                entry["idle"] = idle_text
                expr = run_expression(session, model_config)
                if expr:
                    entry["gaze"] = expr.gaze
                    entry["gaze_type"] = expr.gaze_type
                    entry["expression"] = expr.expression
            log.append(entry)
            return

    # 3. Deliver the beat
    response = deliver_beat(session, beat, label, voice_io=voice_io)
    entry["response"] = response

    # 4. Expression post-processing
    expr = run_expression(session, model_config)
    if expr:
        entry["gaze"] = expr.gaze
        entry["gaze_type"] = expr.gaze_type
        entry["expression"] = expr.expression

    # 5. Advance
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    script.advance()
    if script.current_beat:
        console.print(f"[magenta]  next beat:  [{script.current_beat.intent}][/magenta]")
    else:
        # Script exhausted — background replan
        console.print("[dim]  script complete, replanning...[/dim]")
        _start_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)

    # 7. Background eval + director
    _bump_version()
    _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
    _start_director(scenario, world, people, session, eval_model_config)

    log.append(entry)


def handle_world_change(session, world, goals, script, change_text, label, model_config, big_model_config, eval_model_config, log, voice_io=None, people=None, scenario=None):
    """Process a /world <description> command. Fast path: immediate reaction, background reconcile."""
    # Store as pending
    world.add_pending(change_text)

    # Inject narrator message for fast path
    narrator_msg = format_pending_narrator(change_text)
    console.print(f"[dim]{narrator_msg}[/dim]")
    session.inject_system(narrator_msg)

    # Character reacts (no synchronous eval — narrator injection provides enough context)
    response = stream_response(session, label, "[React to what just happened.]", voice_io=voice_io, expr_model_config=model_config)
    expr = stream_response._last_expr

    entry = {
        "type": "world",
        "input": change_text,
        "narrator": narrator_msg,
        "response": response,
    }
    if expr:
        entry["gaze"] = expr.gaze
        entry["gaze_type"] = expr.gaze_type
        entry["expression"] = expr.expression
    log.append(entry)

    # Start background reconciliation
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    _start_reconcile(world, model_config, big_model_config, people=people)

    # Background eval + director
    _bump_version()
    _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
    _start_director(scenario, world, people, session, eval_model_config)


def handle_perception(session, world, goals, script, people, scenario, see_text, label, model_config, big_model_config, eval_model_config, log, voice_io=None):
    """Process a /see <text> command. Perception event: narrator injection, character reaction, background reconcile+eval+director."""
    event = PerceptionEvent(description=see_text, source="manual")
    _, narrator_msg = process_perception(event, people, world)

    console.print(f"[dim]{narrator_msg}[/dim]")
    session.inject_system(narrator_msg)

    # Character reacts
    response = stream_response(session, label, "[React to what you just noticed.]", voice_io=voice_io, expr_model_config=model_config)
    expr = stream_response._last_expr

    entry = {
        "type": "see",
        "input": see_text,
        "narrator": narrator_msg,
        "response": response,
    }
    if expr:
        entry["gaze"] = expr.gaze
        entry["gaze_type"] = expr.gaze_type
        entry["expression"] = expr.expression
    log.append(entry)

    # Start background reconciliation + eval + director
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    _start_reconcile(world, model_config, big_model_config, people=people)
    _bump_version()
    _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
    _start_director(scenario, world, people, session, eval_model_config)


def run_sim(sim_name, character, session, world, goals, script, people, scenario, label, model_config, big_model_config, eval_model_config, log, voice_io=None):
    """Run a sim script — replay perception events and dialogue with timing."""
    try:
        sim = load_sim_script(character, sim_name)
    except FileNotFoundError:
        console.print(f"[red]Sim script not found: {sim_name}[/red]")
        return

    console.print(f"[cyan]Sim: {sim_name} ({len(sim.events)} events)[/cyan]")
    start_time = time.time()

    for i, event in enumerate(sim.events):
        # Wait for the right time offset
        elapsed = time.time() - start_time
        wait = event.time_offset - elapsed
        if wait > 0:
            time.sleep(wait)

        # Check background results between events
        stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
        if world is not None:
            _check_reconcile(world, log, people=people)
        needs_plan, plan_request = _check_eval(session, script, log)
        if needs_plan:
            _start_plan(session, world, goals, plan_request, big_model_config, people=people, stage_goal=stage_goal)
        _check_director(scenario, script, session, world, goals, people, big_model_config)
        _check_plan(script)

        console.print(f"[cyan]Sim: event {i + 1}/{len(sim.events)}[/cyan]")

        desc = event.description
        if desc.startswith('"') and desc.endswith('"'):
            # Quoted line → user message
            text = desc.strip('"')
            console.print(f"[bold blue]You[/bold blue]: {text}")
            response = stream_response(session, label, text, voice_io=voice_io, expr_model_config=model_config)
            expr = stream_response._last_expr
            entry = {"type": "send", "input": text, "response": response}
            if expr:
                entry["gaze"] = expr.gaze
                entry["gaze_type"] = expr.gaze_type
                entry["expression"] = expr.expression
            log.append(entry)
            _bump_version()
            _start_eval(session, world, goals, script, eval_model_config, people=people, stage_goal=stage_goal)
            _start_director(scenario, world, people, session, eval_model_config)
        else:
            # Unquoted → perception event
            handle_perception(session, world, goals, script, people, scenario, desc, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io)

    console.print(f"[cyan]Sim: complete[/cyan]")


def stream_response(session, label, message, voice_io=None, expr_model_config=None) -> str:
    """Send a message and stream the response. Returns full response text.

    When voice_io is provided, also feeds chunks to TTS and checks for barge-in.
    When expr_model_config is provided, fires expression_call as soon as LLM
    streaming finishes (overlaps with TTS playback). Result stored on
    stream_response._last_expr for the caller to pick up.
    """
    stream_response._last_expr = None
    npc_name = Text(f"{label}: ", style="bold magenta")
    console.print(npc_name, end="")
    full_response = []
    t_start = time.time()
    t_first = None
    gen = session.send(message)

    if voice_io is not None:
        # Voice mode: feed chunks to TTS, check for barge-in
        voice_io._cancelled.clear()
        voice_io._is_speaking = True
        if voice_io._speaker is not None:
            voice_io._speaker.reset_counters()
            voice_io._speaker._audio_started.clear()

        for chunk in gen:
            if voice_io._cancelled.is_set():
                break
            if t_first is None:
                t_first = time.time()
            full_response.append(chunk)
            console.print(chunk, end="", highlight=False)
            voice_io._tts.send_text(chunk)

        t_llm_done = time.time()

        # Fire expression as soon as LLM is done (overlaps with TTS playback)
        if expr_model_config is not None and full_response:
            line_text = "".join(full_response)
            stream_response._last_expr = run_expression(session, expr_model_config, line=line_text)

        # Signal end of text input
        if not voice_io._cancelled.is_set() and voice_io._tts is not None:
            voice_io._tts.flush()

        # Wait for ElevenLabs to finish generating
        if not voice_io._cancelled.is_set() and voice_io._tts is not None:
            voice_io._tts.wait_for_done(timeout=15.0)

        # Wait for first audio to arrive at the speaker
        t_first_audio = None
        if not voice_io._cancelled.is_set() and voice_io._speaker is not None:
            if voice_io._speaker.wait_for_audio(timeout=0.5):
                t_first_audio = time.time()

        # Wait for speaker to finish playback
        if voice_io._speaker is not None and not voice_io._cancelled.is_set():
            while not voice_io._speaker.is_done():
                if voice_io._cancelled.is_set():
                    break
                time.sleep(0.05)

        # Close TTS for next utterance
        if voice_io._tts is not None:
            voice_io._tts.close()
        voice_io._is_speaking = False
        voice_io._tts_done_at = time.time()
    else:
        for chunk in gen:
            if t_first is None:
                t_first = time.time()
            full_response.append(chunk)
            console.print(chunk, end="", highlight=False)

        # Fire expression after LLM streaming (no TTS to overlap with, but still before timing print)
        if expr_model_config is not None and full_response:
            line_text = "".join(full_response)
            stream_response._last_expr = run_expression(session, expr_model_config, line=line_text)

    # Close generator to trigger chat.py's finally block (records partial response)
    gen.close()

    response_text = "".join(full_response)

    # Truncate history to approximate what was actually spoken on barge-in
    if voice_io is not None and voice_io._barged_in and response_text:
        ratio = voice_io.speaker_playback_ratio
        spoken_chars = max(int(len(response_text) * ratio), 1)
        if spoken_chars < len(response_text):
            truncated = response_text[:spoken_chars] + " \u2014"
            session.replace_last_assistant(truncated)
            response_text = truncated

    # Clear barge-in flag after truncation check (must happen after, not before)
    if voice_io is not None:
        voice_io._barged_in = False

    # Timing info
    t_end = time.time()
    ttft_ms = int((t_first - t_start) * 1000) if t_first else 0
    if voice_io is not None and t_first:
        llm_ms = int((t_llm_done - t_start) * 1000)
        first_audio_ms = int((t_first_audio - t_start) * 1000) if t_first_audio else 0
        spoken_ms = int((t_end - t_start) * 1000)
        console.print(f"\n[dim]  {ttft_ms}ms TTFT · {llm_ms}ms LLM · {first_audio_ms}ms first audio · {spoken_ms}ms spoken[/dim]\n")
    else:
        total_ms = int((t_end - t_start) * 1000)
        console.print(f"\n[dim]  {ttft_ms}ms TTFT · {total_ms}ms total[/dim]\n")
    return response_text


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


def show_help(voice_active: bool = False):
    help_text = (
        "/info           - Show all state (world, plan, goals, stage, people)\n"
        "/world <text>   - Describe a change to the world\n"
        "/beat           - Deliver next scripted beat (time passes)\n"
        "/see <text>     - Describe what the character sees (perception event)\n"
        "/sim <name>     - Run a sim script (timed perception events)\n"
        "1-4             - Trigger a stage exit (shown in HUD)\n"
        "/voice          - Toggle voice mode on/off\n"
        "/devices        - List audio input/output devices\n"
        "/reload         - Reload prompt files, restart conversation\n"
        "/trace          - Show system prompt, model, token usage\n"
        "/back           - Return to character selection\n"
        "/quit           - Exit program\n"
        "Ctrl+C          - Return to character selection"
    )
    if voice_active:
        help_text += (
            "\n\n[bold]Voice Hotkeys[/bold]\n"
            "i               - /info\n"
            "b               - /beat\n"
            "t               - /trace\n"
            "1-4             - Stage triggers\n"
            "Escape          - Toggle voice off\n"
            "q               - Quit"
        )
    console.print(Panel(help_text, title="Commands", border_style="dim"))


def run_smoke():
    """Smoke test: auto-select greg, send a message, /world change, /beat, exit.

    Exercises the full startup + chat path with real LLM calls.
    Returns exit code 0 on success, 1 on failure.
    """
    console.print(Panel("[bold]Smoke Test[/bold]", border_style="yellow"))
    model_config = get_chat_model_config()
    if model_config is None:
        chat_cfg = MODELS.get(CHAT_MODEL, {})
        console.print(f"[red]Smoke test failed: chat model unavailable — set {chat_cfg.get('api_key_env', 'API key')} in .env[/red]")
        return 1

    character = "greg"
    label = character.replace("_", " ").title()
    big_model_config = get_big_model_config()
    eval_model_config = big_model_config if big_model_config else model_config
    log: list[dict] = []

    console.print(f"[dim]Chat: {model_config['name']}[/dim]")
    if big_model_config:
        console.print(f"[dim]Big model: {MODELS[BIG_MODEL]['name']}[/dim]")

    # --- Startup: load everything ---
    world = load_world_state(character)
    goals = load_goals(character)
    people = PeopleState()
    scenario = load_scenario_script(character)
    system_prompt = load_prompt(character, world_state=world, people_state=people)
    session = ChatSession(system_prompt, model_config)
    script = Script()

    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    if scenario:
        console.print(f"[dim]Scenario: {scenario.name} (stage: {scenario.current_stage})[/dim]")

    console.print("[dim]Planning...[/dim]")
    plan_result = run_plan(session, world, goals, "", big_model_config, people=people, stage_goal=stage_goal)
    if plan_result and plan_result.beats:
        apply_plan(script, plan_result)
        console.print(f"[dim]Got {len(script.beats)} beats[/dim]")

    # --- Scripted sequence ---
    steps = [
        ("send", "Hello, what's going on here?"),
        ("world", "a bird lands on the desk"),
        ("beat", None),
    ]

    for action, text in steps:
        console.print(f"\n[yellow]▶ smoke: {action}" + (f" {text}" if text else "") + "[/yellow]")
        try:
            if action == "send":
                response = stream_response(session, label, text)
                log.append({"type": "send", "input": text, "response": response})
                _bump_version()
                if not response.strip():
                    console.print("[red]Smoke test failed: empty response[/red]")
                    return 1
            elif action == "world":
                handle_world_change(session, world, goals, script, text, label,
                                    model_config, big_model_config, eval_model_config, log,
                                    people=people, scenario=scenario)
            elif action == "beat":
                handle_beat(session, world, goals, script, label,
                            model_config, big_model_config, eval_model_config, log,
                            people=people, scenario=scenario)
        except Exception as e:
            console.print(f"[red]Smoke test failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 1

    console.print(f"\n[bold green]Smoke test passed[/bold green] ({len(log)} log entries)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="NPC Character Chat")
    parser.add_argument("--voice", action="store_true", help="Start in voice mode (Deepgram STT + ElevenLabs TTS)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (auto greg, scripted inputs, exit)")
    args = parser.parse_args()

    if args.smoke:
        sys.exit(run_smoke())

    cfg = load_config()

    # --voice flag overrides config; config.voice.enabled is the default
    voice_mode = args.voice or cfg.voice.enabled

    console.print(Panel("[bold]NPC Character Chat[/bold]", border_style="green"))
    model_config = get_chat_model_config()
    if model_config is None:
        chat_cfg = MODELS.get(CHAT_MODEL, {})
        console.print(f"[red]Chat model unavailable — set {chat_cfg.get('api_key_env', 'API key')} in .env[/red]")
        return
    console.print(f"[dim]Chat: {model_config['name']}[/dim]")
    while True:
        character = pick_character()
        if character is None:
            console.print("Goodbye!")
            break
        chat_loop(character, model_config, voice_mode=voice_mode, voice_cfg=cfg.voice)


if __name__ == "__main__":
    main()
