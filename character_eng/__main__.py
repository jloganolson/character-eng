import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.chat import ChatSession
from character_eng.config import load_config
from character_eng.models import BIG_MODEL, CHAT_MODEL, MODELS
from character_eng.utils import ts as _ts
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
    ScriptCheckResult,
    ThoughtResult,
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
    script_check_call,
    single_beat_call,
    thought_call,
)

console = Console()

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# --- Dashboard ---
_collector = None  # set in chat_loop / main when dashboard is active
_dashboard_input_queue = None


def _push(event_type: str, data: dict) -> None:
    """Push an event to the dashboard collector (no-op if dashboard is off)."""
    if _collector is not None:
        _collector.push(event_type, data)


def _get_input(session_id: str) -> str:
    """Get user input from console or dashboard queue.

    When dashboard is active, starts a thread to read stdin so that both
    stdin and the dashboard input queue can be polled.
    """
    import queue as _queue

    if _dashboard_input_queue is None:
        return console.input(f"[bold blue]You[/bold blue] [dim]{session_id}[/dim]: ").strip()

    # Both sources feed a merge queue
    merge_q: _queue.Queue = _queue.Queue()

    def _stdin_reader():
        try:
            line = console.input(f"[bold blue]You[/bold blue] [dim]{session_id}[/dim]: ").strip()
            merge_q.put(("stdin", line))
        except (EOFError, KeyboardInterrupt) as e:
            merge_q.put(("error", e))

    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    while True:
        # Check dashboard queue
        try:
            text = _dashboard_input_queue.get(timeout=0.1)
            console.print(f"[bold blue]You[/bold blue] [dim]{session_id}[/dim]: {text}")
            return text
        except _queue.Empty:
            pass
        # Check stdin
        try:
            source, value = merge_q.get_nowait()
            if source == "error":
                raise value
            return value
        except _queue.Empty:
            pass

# --- Background reconciliation threading ---
_reconcile_lock = threading.Lock()
_reconcile_result: WorldUpdate | None = None
_reconcile_thread: threading.Thread | None = None

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


def _start_reconcile(world, model_config, big_model_config=None, people=None):
    """Drain pending changes and spawn background reconcile thread."""
    global _reconcile_thread
    pending = world.clear_pending()
    if not pending:
        return
    # 8B handles reconcile (86% vs 88% at 70B, experiment 5)
    cfg = model_config
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

    _push("reconcile", {
        "remove_facts": result.remove_facts,
        "add_facts": result.add_facts,
        "events": result.events,
    })
    _push("world_state", {
        "static_facts": [{"id": "", "text": s} for s in (world.static or [])],
        "dynamic_facts": [{"id": k, "text": v} for k, v in (world.dynamic or {}).items()],
        "pending": list(world.pending),
    })
    if people is not None:
        _push("people_state", {"people": [
            {"id": p.person_id, "name": p.name, "facts": [{"id": k, "text": v} for k, v in p.facts.items()]}
            for p in people.people.values()
        ]})


# --- Sync eval microservices (8B) ---

def run_eval_sync(session, world, script, model_config, log, people=None, stage_goal=""):
    """Run script_check + thought synchronously on chat model. Returns (needs_plan, plan_request).

    Replaces the old background eval thread. Both calls use 8B for speed.
    """
    # Script check: advance/hold/off_book classification
    beat = script.current_beat if script else None
    if beat is None:
        return (False, "")

    try:
        check = script_check_call(
            beat=beat,
            history=session.get_history(),
            model_config=model_config,
            world=world,
        )
    except Exception as e:
        console.print(f"[dim]  script_check error: {e}[/dim]")
        return (False, "")

    # Thought: inner monologue
    try:
        tresult = thought_call(
            system_prompt=session.system_prompt,
            history=session.get_history(),
            model_config=model_config,
        )
    except Exception as e:
        console.print(f"[dim]  thought error: {e}[/dim]")
        tresult = ThoughtResult(thought="")

    # Build an EvalResult-compatible representation for display/logging
    eval_compat = EvalResult(
        thought=tresult.thought,
        script_status=check.status,
        plan_request=check.plan_request,
    )
    display_eval(eval_compat)
    inject_eval(session, eval_compat)
    log.append({"type": "eval", **eval_to_dict(eval_compat)})

    if check.status == "advance":
        script.advance()
        if script.current_beat:
            console.print(f"[magenta]  {_ts()} next beat:  [{script.current_beat.intent}][/magenta]")
            return (False, "")
        else:
            console.print(f"[dim]  {_ts()} script complete, replanning...[/dim]")
            return (True, "")
    elif check.status == "off_book" and check.plan_request:
        return (True, check.plan_request)

    return (False, "")


# --- Parallel post-response microservices ---

def run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal="", expression_line=None, vision_mgr=None):
    """Run eval (script_check + thought) and director in parallel. Returns (needs_plan, plan_request, expr_result).

    When expression_line is provided, also runs expression_call in the parallel batch.
    ~350ms total instead of ~1050ms sequential.
    """
    beat = script.current_beat if script else None
    gaze_targets = vision_mgr.get_gaze_targets() if vision_mgr else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        # Fire all LLM calls concurrently
        fut_check = pool.submit(script_check_call, beat=beat, history=session.get_history(),
                                model_config=model_config, world=world) if beat else None
        fut_thought = pool.submit(thought_call, system_prompt=session.system_prompt,
                                  history=session.get_history(), model_config=model_config) if beat else None
        fut_director = pool.submit(director_call, scenario=scenario, world=world, people=people,
                                   history=session.get_history(), model_config=model_config) if scenario else None
        fut_expr = pool.submit(expression_call, expression_line, model_config, gaze_targets) if expression_line else None

        # Gather results
        check = None
        tresult = ThoughtResult()
        dir_result = None
        expr_result = None

        if fut_check:
            try:
                check = fut_check.result()
            except Exception as e:
                console.print(f"[dim]  script_check error: {e}[/dim]")

        if fut_thought:
            try:
                tresult = fut_thought.result()
            except Exception as e:
                console.print(f"[dim]  thought error: {e}[/dim]")

        if fut_director:
            try:
                dir_result = fut_director.result()
            except Exception as e:
                console.print(f"[dim]  director error: {e}[/dim]")

        if fut_expr:
            try:
                expr_result = fut_expr.result()
            except Exception as e:
                console.print(f"[dim]  expression error: {e}[/dim]")

    # Display + inject expression result
    if expr_result and (expr_result.gaze or expr_result.expression):
        gaze_label = expr_result.gaze
        if expr_result.gaze_type == "glance":
            gaze_label += " (glance)"
        console.print(f"  {_ts()} gaze: {gaze_label}  expression: {expr_result.expression}")
        _push("expression", {
            "gaze": expr_result.gaze, "gaze_type": expr_result.gaze_type,
            "expression": expr_result.expression,
        })
        parts = []
        if expr_result.gaze:
            if expr_result.gaze_type == "glance":
                parts.append(f"[glance:{expr_result.gaze}]")
            else:
                parts.append(f"[gaze:{expr_result.gaze}]")
        if expr_result.expression:
            parts.append(f"[emote:{expr_result.expression}]")
        if parts:
            session.inject_system(" ".join(parts))

    # Process eval result
    needs_plan = False
    plan_request = ""
    if check is not None:
        eval_compat = EvalResult(
            thought=tresult.thought,
            script_status=check.status,
            plan_request=check.plan_request,
        )
        display_eval(eval_compat)
        inject_eval(session, eval_compat)
        log.append({"type": "eval", **eval_to_dict(eval_compat)})
        _push("eval", {
            "thought": tresult.thought, "script_status": check.status,
            "plan_request": check.plan_request,
        })

        if check.status == "advance":
            script.advance()
            if script.current_beat:
                console.print(f"[magenta]  {_ts()} next beat:  [{script.current_beat.intent}][/magenta]")
                _push("beat_advance", {"next_intent": script.current_beat.intent})
            else:
                console.print(f"[dim]  {_ts()} script complete, replanning...[/dim]")
                _push("beat_advance", {"script_complete": True})
                needs_plan = True
        elif check.status == "off_book" and check.plan_request:
            needs_plan = True
            plan_request = check.plan_request

    # Process director result
    if dir_result is not None:
        if dir_result.status == "advance" and dir_result.exit_index >= 0:
            if scenario and scenario.active_stage:
                stage = scenario.active_stage
                if 0 <= dir_result.exit_index < len(stage.exits):
                    exit_obj = stage.exits[dir_result.exit_index]
                    if scenario.advance_to(exit_obj.goto):
                        new_stage = scenario.active_stage
                        console.print(f"[cyan]  {_ts()} director: {dir_result.thought}[/cyan]")
                        _push("director", {
                            "thought": dir_result.thought, "status": "advance",
                            "exit_index": dir_result.exit_index,
                            "new_stage": new_stage.name if new_stage else None,
                        })
                        if new_stage:
                            console.print(f"[bold cyan]{_ts()} Stage → {new_stage.name}: {new_stage.goal}[/bold cyan]")
                            _push("stage_change", {
                                "old_stage": stage.name, "new_stage": new_stage.name,
                                "new_goal": new_stage.goal,
                            })
                            result = single_beat_call(
                                system_prompt=session.system_prompt, world=world,
                                history=session.get_history(), goals=goals, model_config=model_config,
                                people=people, stage_goal=new_stage.goal,
                            )
                            if result.beats:
                                apply_plan(script, result)
        else:
            if dir_result.thought:
                console.print(f"[dim]  {_ts()} director: {dir_result.thought}[/dim]")
            _push("director", {
                "thought": dir_result.thought or "", "status": dir_result.status or "hold",
                "exit_index": dir_result.exit_index,
            })

    return needs_plan, plan_request, expr_result


# --- Sync director (8B microservice) ---

def run_director_sync(scenario, world, people, session, model_config, script, goals, big_model_config=None):
    """Run director synchronously on the chat model. Returns whether a stage transition occurred."""
    if scenario is None:
        return False
    try:
        result = director_call(scenario, world, people, session.get_history(), model_config)
    except Exception as e:
        console.print(f"[dim]  director error: {e}[/dim]")
        return False

    if result.status == "advance" and result.exit_index >= 0:
        stage = scenario.active_stage
        if stage and 0 <= result.exit_index < len(stage.exits):
            exit_obj = stage.exits[result.exit_index]
            if scenario.advance_to(exit_obj.goto):
                new_stage = scenario.active_stage
                console.print(f"[cyan]  {_ts()} director: {result.thought}[/cyan]")
                if new_stage:
                    console.print(f"[bold cyan]{_ts()} Stage → {new_stage.name}: {new_stage.goal}[/bold cyan]")
                    plan_result = single_beat_call(
                        system_prompt=session.system_prompt, world=world,
                        history=session.get_history(), goals=goals, model_config=model_config,
                        people=people, stage_goal=new_stage.goal,
                    )
                    if plan_result.beats:
                        apply_plan(script, plan_result)
                return True
    else:
        if result.thought:
            console.print(f"[dim]  {_ts()} director: {result.thought}[/dim]")

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


def handle_trigger(trigger_num, scenario, session, world, goals, script, people, label, model_config, big_model_config, eval_model_config, log, voice_io=None, vision_mgr=None):
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
    _push("stage_change", {
        "old_stage": old_stage, "new_stage": new_stage.name,
        "new_goal": new_stage.goal if new_stage else "",
    })

    # 2. Inject exit condition as perception event → character reacts
    handle_perception(
        session, world, goals, script, people, scenario,
        exit_obj.condition, label,
        model_config, big_model_config, eval_model_config, log,
        voice_io=voice_io, vision_mgr=vision_mgr,
    )

    # 3. Clear stale script and synchronously replan with new stage goal
    #    Must be synchronous — if async, /beat would deliver stale beats from the old stage.
    if voice_io is not None:
        voice_io._cancel_auto_beat()
    script.replace([])
    stage_goal = new_stage.goal if new_stage else ""
    plan_result = run_plan(session, world, goals, "", model_config, people=people, stage_goal=stage_goal)
    if plan_result and plan_result.beats:
        apply_plan(script, plan_result)

    # Update visual focus for new stage
    if vision_mgr is not None:
        vision_mgr.update_context(beat=script.current_beat, stage_goal=stage_goal)
        vision_mgr.update_focus(
            beat=script.current_beat, stage_goal=stage_goal, thought="",
            world=world, people=people, model_config=model_config,
        )


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


def chat_loop(character: str, model_config: dict, voice_mode: bool = False, voice_cfg=None,
              vision_mode: bool = False, vision_cfg=None, auto_sim: str | None = None):
    """Run the chat loop for a character. Returns on /back or Ctrl+C."""
    label = character.replace("_", " ").title()
    session_id = uuid.uuid4().hex[:8]
    log: list[dict] = []
    # All LLM calls now use 8B chat model (parallel microservices, single-beat planning)
    big_model_config = get_big_model_config()  # kept for API compat
    eval_model_config = model_config  # kept for API compat

    # --- Vision setup ---
    vision_mgr = None
    if vision_mode and vision_cfg:
        from character_eng.vision.manager import VisionManager
        vision_mgr = VisionManager(
            service_url=vision_cfg.service_url,
            min_interval=vision_cfg.synthesis_min_interval,
        )

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

        # Start vision manager (after people setup)
        if vision_mgr is not None:
            vision_mgr.start(model_config, world=world, people=people, collector=_collector)
            vision_mgr.update_context(stage_goal=stage_goal)
            console.print("[green]Vision active[/green]")
            # Initial focus
            vision_mgr.update_focus(
                beat=None, stage_goal=stage_goal, thought="",
                world=world, people=people, model_config=model_config,
            )

        # Generate initial script at boot (synchronous, 8B single-beat)
        plan_result = run_plan(session, world, goals, "", model_config, people=people, stage_goal=stage_goal)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)

        had_user_input = False
        _arm_auto_beat = True  # arm on startup so first beat fires without user input

        mode_tag = " [green]voice[/green]" if voice_io is not None else ""
        console.print(f"\n[bold green]Chatting with {label} ({model_config['name']})[/bold green]{mode_tag}  [dim]{session_id}[/dim]  (type /help for commands)\n")

        model_key = next((k for k, v in MODELS.items() if v is model_config), "unknown")
        _push("session_start", {
            "character": label, "model": model_key, "session_id": session_id,
            "stage": scenario.active_stage.name if scenario and scenario.active_stage else "",
            "goal": stage_goal,
        })
        # Push initial world/people/stage state
        if world is not None:
            _push("world_state", {
                "static_facts": [{"id": "", "text": s} for s in (world.static or [])],
                "dynamic_facts": [{"id": k, "text": v} for k, v in (world.dynamic or {}).items()],
                "pending": [],
            })
        if scenario and scenario.active_stage:
            _push("stage_change", {
                "old_stage": "", "new_stage": scenario.active_stage.name,
                "new_goal": scenario.active_stage.goal,
            })

        # Auto-sim: run sim script then exit
        if auto_sim:
            run_sim(auto_sim, character, session, world, goals, script, people, scenario, label,
                    model_config, big_model_config, eval_model_config, log, vision_mgr=vision_mgr)
            save_chat_log(character, model_config, log, session_id)
            save_chat_html(character, model_config, log, session_id)
            return

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
                    console.print(f"[dim]  {_ts()} → {user_input}[/dim]")
                elif user_input in ("1", "2", "3", "4"):
                    console.print(f"[dim]  {_ts()} → trigger {user_input}[/dim]")
                else:
                    console.print(f"[bold blue]You[/bold blue] [dim]{session_id} {_ts()}[/dim]: {user_input}")
            else:
                try:
                    user_input = _get_input(session_id)
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

            # Drain vision events
            if vision_mgr is not None:
                for event in vision_mgr.drain_events():
                    _, narrator_msg = process_perception(event, people, world)
                    console.print(f"[dim]{narrator_msg}[/dim]")
                    session.inject_system(narrator_msg)
                    _push("perception", {"source": event.source, "description": event.description})
                vision_mgr.update_context(world=world, people=people, stage_goal=stage_goal)

            # --- Command dispatch ---
            cmd = user_input.lower()
            if cmd == "/quit":
                if voice_io is not None:
                    voice_io.stop()
                    voice_io = None
                _push("session_end", {"total_turns": len([e for e in log if e.get("type") in ("send", "beat", "world", "see")])})
                save_chat_log(character, model_config, log, session_id)
                save_chat_html(character, model_config, log, session_id)
                console.print("Goodbye!")
                sys.exit(0)
            elif cmd == "/back":
                if voice_io is not None:
                    voice_io.stop()
                    voice_io = None
                _push("session_end", {"total_turns": len([e for e in log if e.get("type") in ("send", "beat", "world", "see")])})
                save_chat_log(character, model_config, log, session_id)
                save_chat_html(character, model_config, log, session_id)
                return
            elif cmd == "/reload":
                console.print("[yellow]Reloading prompts...[/yellow]")
                if vision_mgr is not None:
                    vision_mgr.stop()
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
            elif cmd == "/vision":
                if vision_mgr is not None:
                    ctx_text = vision_mgr.context.render_for_prompt()
                    if ctx_text:
                        console.print(Panel(ctx_text, title="Vision Context", border_style="cyan"))
                    else:
                        console.print("[dim]No visual data yet.[/dim]")
                    targets = vision_mgr.get_gaze_targets()
                    if targets:
                        console.print(f"[dim]Gaze targets: {', '.join(targets)}[/dim]")
                else:
                    console.print("[dim]Vision not active. Use --vision flag.[/dim]")
                continue
            elif cmd == "/beat":
                handle_beat(session, world, goals, script, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, people=people, scenario=scenario, vision_mgr=vision_mgr)
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
                handle_world_change(session, world, goals, script, change_text, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, people=people, scenario=scenario, vision_mgr=vision_mgr)
                _arm_auto_beat = True
                continue
            elif cmd.startswith("/see "):
                if world is None:
                    console.print("[dim]No world state for this character.[/dim]")
                    continue
                see_text = user_input[5:]  # preserve original case
                handle_perception(session, world, goals, script, people, scenario, see_text, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, vision_mgr=vision_mgr)
                _arm_auto_beat = True
                continue
            elif cmd.startswith("/sim "):
                sim_name = user_input[5:].strip()
                if not sim_name:
                    console.print("[red]Usage: /sim <name>[/red]")
                    continue
                run_sim(sim_name, character, session, world, goals, script, people, scenario, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, vision_mgr=vision_mgr)
                _arm_auto_beat = True
                continue
            elif cmd in ("1", "2", "3", "4"):
                handle_trigger(int(cmd), scenario, session, world, goals, script, people, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, vision_mgr=vision_mgr)
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
            _push("turn_start", {"input_type": "send", "input_text": user_input})

            # First user input: natural LLM response, then parallel eval + replan
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
                needs_plan, plan_request, _ = run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr)
                # Always replan after first user input
                result = single_beat_call(
                    system_prompt=session.system_prompt, world=world,
                    history=session.get_history(), goals=goals, model_config=model_config,
                    plan_request=plan_request, people=people, stage_goal=stage_goal,
                )
                if result.beats:
                    apply_plan(script, result)
                _arm_auto_beat = True
                continue

            # Bootstrap: if no script, plan synchronously before responding
            if script.is_empty():
                result = single_beat_call(
                    system_prompt=session.system_prompt, world=world,
                    history=session.get_history(), goals=goals, model_config=model_config,
                    people=people, stage_goal=stage_goal,
                )
                if result.beats:
                    apply_plan(script, result)

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
            needs_plan, plan_request, _ = run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr)
            if needs_plan:
                result = single_beat_call(
                    system_prompt=session.system_prompt, world=world,
                    history=session.get_history(), goals=goals, model_config=model_config,
                    plan_request=plan_request, people=people, stage_goal=stage_goal,
                )
                if result.beats:
                    apply_plan(script, result)
            _arm_auto_beat = True


def run_eval(session, world, goals, script, model_config):
    """Run eval_call synchronously, return EvalResult or None on error."""
    console.print(f"[dim]{_ts()} Evaluating...[/dim]")
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


def run_plan(session, world, goals, plan_request, model_config, people=None, stage_goal=""):
    """Run single_beat_call synchronously, return PlanResult or None on error."""
    if model_config is None:
        console.print("[dim]Planner unavailable, skipping.[/dim]")
        return None
    console.print(f"[dim]{_ts()} Planning...[/dim]")
    try:
        return single_beat_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            goals=goals,
            model_config=model_config,
            plan_request=plan_request,
            people=people,
            stage_goal=stage_goal,
        )
    except Exception as e:
        console.print(f"[red]Plan error: {e}[/red]")
        return None


def display_eval(result):
    """Display all eval fields."""
    console.print(f"[dim]  {_ts()} thought:    {result.thought}[/dim]")
    console.print(f"[dim]  {_ts()} status:     {result.script_status}[/dim]")
    if result.plan_request:
        console.print(f"[dim]  {_ts()} plan_req:   {result.plan_request}[/dim]")


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
        console.print(f"  {_ts()} gaze: {gaze_label}  expression: {result.expression}")
        _push("expression", {
            "gaze": result.gaze, "gaze_type": result.gaze_type,
            "expression": result.expression,
        })

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
    npc_name = Text(f"{_ts()} {label}: ", style="bold magenta")
    console.print(npc_name, end="")
    console.print(beat.line)
    console.print()
    _push("response_chunk", {"text": beat.line})
    _push("response_done", {"full_text": beat.line, "ttft_ms": 0, "total_ms": 0})
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
    console.print(f"[magenta]{_ts()} Script loaded ({len(plan_result.beats)} beats):[/magenta]")
    for i, beat in enumerate(plan_result.beats):
        marker = "→" if i == 0 else " "
        console.print(f"[magenta]  {marker} {i}. [{beat.intent}] \"{beat.line}\"[/magenta]")
    _push("plan", {"beats": [{"intent": b.intent, "line": b.line} for b in plan_result.beats]})


def handle_beat(session, world, goals, script, label, model_config, big_model_config, eval_model_config, log, voice_io=None, people=None, scenario=None, vision_mgr=None):
    """Process a /beat command — condition-gated beat delivery (time passes)."""
    _push("turn_start", {"input_type": "beat", "input_text": ""})
    entry = {"type": "beat"}

    # 1. If no current beat, replan (synchronous — user explicitly asked for a beat)
    if script.current_beat is None:
        plan_result = run_plan(session, world, goals, "", model_config)
        if plan_result and plan_result.beats:
            apply_plan(script, plan_result)
        if script.current_beat is None:
            console.print("[dim]No beats available.[/dim]\n")
            log.append(entry)
            return

    beat = script.current_beat

    # 2. If beat has a condition, check it (synchronous — gates delivery)
    if beat.condition:
        console.print(f"[dim]{_ts()} Checking condition: {beat.condition}[/dim]")
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

    # 4. Advance
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    script.advance()
    if script.current_beat:
        console.print(f"[magenta]  {_ts()} next beat:  [{script.current_beat.intent}][/magenta]")
    else:
        console.print(f"[dim]  {_ts()} script complete[/dim]")

    # 5. Parallel: expression + eval + director
    _bump_version()
    needs_plan, plan_request, expr = run_post_response(
        session, world, script, model_config, log, scenario, people, goals, stage_goal,
        expression_line=response, vision_mgr=vision_mgr,
    )
    if expr:
        entry["gaze"] = expr.gaze
        entry["gaze_type"] = expr.gaze_type
        entry["expression"] = expr.expression

    # 6. Replan if needed (script complete or off_book)
    if needs_plan or not script.current_beat:
        result = single_beat_call(
            system_prompt=session.system_prompt, world=world,
            history=session.get_history(), goals=goals, model_config=model_config,
            plan_request=plan_request, people=people, stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)

    log.append(entry)


def handle_world_change(session, world, goals, script, change_text, label, model_config, big_model_config, eval_model_config, log, voice_io=None, people=None, scenario=None, vision_mgr=None):
    """Process a /world <description> command. Fast path: immediate reaction, background reconcile."""
    _push("turn_start", {"input_type": "world", "input_text": change_text})
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
    _start_reconcile(world, model_config, people=people)

    # Parallel eval + director
    _bump_version()
    needs_plan, plan_request, _ = run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr)
    if needs_plan:
        result = single_beat_call(
            system_prompt=session.system_prompt, world=world,
            history=session.get_history(), goals=goals, model_config=model_config,
            plan_request=plan_request, people=people, stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)


def handle_perception(session, world, goals, script, people, scenario, see_text, label, model_config, big_model_config, eval_model_config, log, voice_io=None, vision_mgr=None):
    """Process a /see <text> command. Perception event: narrator injection, character reaction, background reconcile+eval+director."""
    _push("turn_start", {"input_type": "see", "input_text": see_text})
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

    # Start background reconciliation + parallel eval + director
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    _start_reconcile(world, model_config, people=people)
    _bump_version()
    needs_plan, plan_request, _ = run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr)
    if needs_plan:
        result = single_beat_call(
            system_prompt=session.system_prompt, world=world,
            history=session.get_history(), goals=goals, model_config=model_config,
            plan_request=plan_request, people=people, stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)


def run_sim(sim_name, character, session, world, goals, script, people, scenario, label, model_config, big_model_config, eval_model_config, log, voice_io=None, vision_mgr=None):
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
            needs_plan, plan_request, _ = run_post_response(session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr)
            if needs_plan:
                result = single_beat_call(
                    system_prompt=session.system_prompt, world=world,
                    history=session.get_history(), goals=goals, model_config=model_config,
                    plan_request=plan_request, people=people, stage_goal=stage_goal,
                )
                if result.beats:
                    apply_plan(script, result)
        else:
            # Unquoted → perception event
            handle_perception(session, world, goals, script, people, scenario, desc, label, model_config, big_model_config, eval_model_config, log, voice_io=voice_io, vision_mgr=vision_mgr)

    console.print(f"[cyan]Sim: complete[/cyan]")


def stream_response(session, label, message, voice_io=None, expr_model_config=None) -> str:
    """Send a message and stream the response. Returns full response text.

    When voice_io is provided, also feeds chunks to TTS and checks for barge-in.
    When expr_model_config is provided, fires expression_call as soon as LLM
    streaming finishes (overlaps with TTS playback). Result stored on
    stream_response._last_expr for the caller to pick up.
    """
    stream_response._last_expr = None
    npc_name = Text(f"{_ts()} {label}: ", style="bold magenta")
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
            _push("response_chunk", {"text": chunk})
            voice_io._tts.send_text(chunk)

        t_llm_done = time.time()

        # Signal end of text input — must happen before expression call
        # so TTS can start generating while expression runs in parallel
        if not voice_io._cancelled.is_set() and voice_io._tts is not None:
            voice_io._tts.flush()

        # Fire expression after flush (overlaps with TTS generation + playback)
        if expr_model_config is not None and full_response:
            line_text = "".join(full_response)
            stream_response._last_expr = run_expression(session, expr_model_config, line=line_text)

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
            _push("response_chunk", {"text": chunk})

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
        _push("response_done", {"full_text": response_text, "ttft_ms": ttft_ms, "total_ms": spoken_ms})
    else:
        total_ms = int((t_end - t_start) * 1000)
        console.print(f"\n[dim]  {ttft_ms}ms TTFT · {total_ms}ms total[/dim]\n")
        _push("response_done", {"full_text": response_text, "ttft_ms": ttft_ms, "total_ms": total_ms})
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
        "/vision         - Show current visual context and gaze targets\n"
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
    big_model_config = get_big_model_config()  # kept for API compat
    eval_model_config = model_config  # kept for API compat
    log: list[dict] = []

    console.print(f"[dim]Chat: {model_config['name']}[/dim]")

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

    console.print(f"[dim]{_ts()} Planning...[/dim]")
    plan_result = run_plan(session, world, goals, "", model_config, people=people, stage_goal=stage_goal)
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


def _launch_mock_vision(replay_file: str, vision_cfg) -> "subprocess.Popen | None":
    """Launch mock vision server with a replay JSON file."""
    import subprocess as _sp
    from character_eng.vision.client import VisionClient

    client = VisionClient(vision_cfg.service_url)
    if client.health():
        console.print(f"[dim]Vision service already running at {vision_cfg.service_url}[/dim]")
        return None

    replay_path = Path(replay_file)
    if not replay_path.exists():
        # Try relative to services/vision/replays/
        replay_path = Path(__file__).resolve().parent.parent / "services" / "vision" / "replays" / replay_file
    if not replay_path.exists():
        console.print(f"[red]Replay file not found: {replay_file}[/red]")
        return None

    mock_script = Path(__file__).resolve().parent.parent / "services" / "vision" / "mock_server.py"
    console.print(f"[dim]Starting mock vision server with {replay_path.name}...[/dim]")
    proc = _sp.Popen(
        ["python3", str(mock_script), str(replay_path), "--port", str(vision_cfg.service_port)],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )

    for _ in range(10):
        if client.health():
            console.print(f"[green]Mock vision server ready ({replay_path.name})[/green]")
            return proc
        if proc.poll() is not None:
            console.print("[red]Mock vision server died during startup[/red]")
            return None
        time.sleep(0.5)

    console.print("[red]Mock vision server did not start within 5s[/red]")
    proc.kill()
    return None


def _launch_vision_service(vision_cfg) -> "subprocess.Popen | None":
    """Auto-launch vision service if not running. Returns subprocess or None."""
    import subprocess as _sp
    from character_eng.vision.client import VisionClient

    client = VisionClient(vision_cfg.service_url)
    if client.health():
        console.print(f"[dim]Vision service already running at {vision_cfg.service_url}[/dim]")
        return None

    if not vision_cfg.auto_launch:
        console.print(f"[yellow]Vision service not running at {vision_cfg.service_url} (auto_launch=false)[/yellow]")
        return None

    services_dir = Path(__file__).resolve().parent.parent / "services" / "vision"
    if not (services_dir / "app.py").exists():
        console.print("[red]Vision service not found at services/vision/app.py[/red]")
        return None

    console.print(f"[dim]Starting vision service on port {vision_cfg.service_port}...[/dim]")
    proc = _sp.Popen(
        ["uv", "run", "--project", str(services_dir), "python", str(services_dir / "app.py"),
         "--port", str(vision_cfg.service_port), "--auto-start-trackers"],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )

    # Wait for health check
    for _ in range(60):
        if client.health():
            console.print(f"[green]Vision service ready at {vision_cfg.service_url}[/green]")
            return proc
        if proc.poll() is not None:
            console.print("[red]Vision service process died during startup[/red]")
            return None
        time.sleep(1)

    console.print("[red]Vision service did not start within 60s[/red]")
    proc.kill()
    return None


def main():
    parser = argparse.ArgumentParser(description="NPC Character Chat")
    parser.add_argument("--voice", action="store_true", help="Start in voice mode (Deepgram STT + ElevenLabs TTS)")
    parser.add_argument("--vision", action="store_true", help="Enable vision (auto-starts vision service)")
    parser.add_argument("--vision-mock", metavar="REPLAY", help="Enable vision with mock server using a replay JSON file")
    parser.add_argument("--character", metavar="NAME", help="Auto-select character (skip menu)")
    parser.add_argument("--sim", metavar="NAME", help="Run a sim script non-interactively then exit")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (auto greg, scripted inputs, exit)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the HTML dashboard")
    args = parser.parse_args()

    if args.smoke:
        sys.exit(run_smoke())

    cfg = load_config()

    # --voice flag overrides config; config.voice.enabled is the default
    # --sim forces text mode (voice doesn't make sense for non-interactive runs)
    voice_mode = False if args.sim else (args.voice or cfg.voice.enabled)
    # --vision or --vision-mock overrides config; config.vision.enabled is the default
    vision_mode = args.vision or args.vision_mock or cfg.vision.enabled

    # --- Dashboard setup ---
    global _collector, _dashboard_input_queue
    dashboard_enabled = cfg.dashboard.enabled and not args.no_dashboard and not args.smoke
    if dashboard_enabled:
        import queue as _queue
        from character_eng.dashboard.events import DashboardEventCollector
        from character_eng.dashboard.server import start_dashboard
        from character_eng.open_report import open_in_browser
        _collector = DashboardEventCollector()
        _dashboard_input_queue = _queue.Queue()
        _, dash_port = start_dashboard(_collector, _dashboard_input_queue, port=cfg.dashboard.port)
        dash_url = f"http://127.0.0.1:{dash_port}/"
        console.print(f"[bold green]Dashboard:[/bold green] {dash_url}")
        open_in_browser(dash_url)

    console.print(Panel("[bold]NPC Character Chat[/bold]", border_style="green"))
    model_config = get_chat_model_config()
    if model_config is None:
        chat_cfg = MODELS.get(CHAT_MODEL, {})
        console.print(f"[red]Chat model unavailable — set {chat_cfg.get('api_key_env', 'API key')} in .env[/red]")
        return
    console.print(f"[dim]Chat: {model_config['name']}[/dim]")

    # Auto-launch vision service (real or mock)
    vision_proc = None
    if vision_mode:
        if args.vision_mock:
            vision_proc = _launch_mock_vision(args.vision_mock, cfg.vision)
        else:
            vision_proc = _launch_vision_service(cfg.vision)

    try:
        while True:
            if args.character:
                character = args.character
            else:
                character = pick_character()
            if character is None:
                console.print("Goodbye!")
                break
            chat_loop(character, model_config, voice_mode=voice_mode, voice_cfg=cfg.voice,
                       vision_mode=vision_mode, vision_cfg=cfg.vision,
                       auto_sim=args.sim)
            if args.sim or args.character:
                break  # non-interactive: run once and exit
    finally:
        if vision_proc is not None:
            console.print("[dim]Stopping vision service...[/dim]")
            vision_proc.kill()
            vision_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
