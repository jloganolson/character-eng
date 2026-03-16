"""Scaffold toggle testing — tests toggle × model permutations with
deterministic scripted conversation.

Answers: which scaffolding does each model actually need?

Usage:
  uv run -m character_eng.qa_toggles --model kimi-k2
  uv run -m character_eng.qa_toggles --models groq-llama-8b,kimi-k2
  uv run -m character_eng.qa_toggles --merge logs/qa_toggles_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from character_eng.chat import ChatSession
from character_eng.models import DEFAULT_MODEL, MODELS
from character_eng.perception import PerceptionEvent, process_perception
from character_eng.person import PeopleState
from character_eng.prompts import load_prompt
from character_eng.scenario import DirectorResult, director_call, load_scenario_script
from character_eng.world import (
    EvalResult,
    Goals,
    PlanResult,
    Script,
    ThoughtResult,
    WorldUpdate,
    condition_check_call,
    eval_call,
    format_pending_narrator,
    load_beat_guide,
    load_goals,
    load_world_state,
    next_beat_call,
    reconcile_call,
    script_check_call,
    thought_call,
)

load_dotenv()

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CHARACTER = "greg"


# ---------------------------------------------------------------------------
# Toggle combos
# ---------------------------------------------------------------------------

@dataclass
class ToggleCombo:
    name: str
    beats: bool
    thinker: bool
    director: bool
    description: str


TOGGLE_COMBOS: list[ToggleCombo] = [
    ToggleCombo("all_on", beats=True, thinker=True, director=True,
                description="Baseline — current behavior"),
    ToggleCombo("thought_off", beats=True, thinker=False, director=True,
                description="Is the inner monologue nudge helping?"),
    ToggleCombo("director_off", beats=True, thinker=True, director=False,
                description="Can the model self-signal stage transitions?"),
    ToggleCombo("beats_off", beats=False, thinker=True, director=True,
                description="Can the model improvise from stage goal alone?"),
    ToggleCombo("beats_director_off", beats=False, thinker=True, director=False,
                description="Both macro + micro pacing removed"),
    ToggleCombo("all_scaffold_off", beats=False, thinker=False, director=False,
                description="Model fully on its own"),
]

# ---------------------------------------------------------------------------
# Scripted conversation (8 turns)
# ---------------------------------------------------------------------------

SCRIPTED_TURNS: list[tuple[str, str]] = [
    ("send", "Hey there, what are you?"),
    ("send", "Tell me more about yourself."),
    ("world", "A child walks past carrying a big red balloon"),
    ("beat", ""),
    ("trigger", "1"),
    ("send", "Actually, I want to talk about something totally different."),
    ("send", "Never mind, let's get back on track."),
    ("send", "Thanks, I should get going. Bye!"),
]


# ---------------------------------------------------------------------------
# Toggle-aware ConversationDriver (adapted from qa_personas.py)
# ---------------------------------------------------------------------------


class ToggleDriver:
    """Runs a scripted conversation with specific toggles applied."""

    def __init__(self, character: str, model_config: dict, toggles: ToggleCombo,
                 micro_model_config: dict | None = None):
        self.character = character
        self.model_config = model_config
        self.micro_config = micro_model_config or model_config
        self.toggles = toggles

        # Core state
        self.world = load_world_state(character)
        self.goals = load_goals(character)
        self.people = PeopleState()
        self.scenario = load_scenario_script(character)
        self.system_prompt = load_prompt(character, world_state=self.world, people_state=self.people)
        self.session = ChatSession(self.system_prompt, model_config)
        self.script = Script()
        self.label = character.replace("_", " ").title()

        # Background reconciliation
        self._reconcile_lock = threading.Lock()
        self._reconcile_result: WorldUpdate | None = None
        self._reconcile_thread: threading.Thread | None = None

        # Conversation log
        self.log: list[dict] = []
        self._had_user_input = False

        # Metrics
        self.stages_advanced = 0
        self.beats_generated = 0
        self.thoughts_generated = 0
        self.errors: list[str] = []

    # --- Toggle checks ---

    def _beats_enabled(self) -> bool:
        return self.toggles.beats

    def _thinker_enabled(self) -> bool:
        return self.toggles.thinker

    def _director_enabled(self) -> bool:
        return self.toggles.director

    # --- Collect response (no streaming display) ---

    def _collect_response(self, message: str) -> str:
        chunks = []
        for chunk in self.session.send(message):
            chunks.append(chunk)
        return "".join(chunks)

    # --- Background reconcile ---

    def _run_reconcile(self, pending: list[str]):
        try:
            result = reconcile_call(self.world, pending, self.micro_config, people=self.people)
            with self._reconcile_lock:
                self._reconcile_result = result
        except Exception:
            pass

    def _start_reconcile(self):
        pending = self.world.clear_pending()
        if not pending:
            return
        self._reconcile_thread = threading.Thread(
            target=self._run_reconcile, args=(pending,), daemon=True
        )
        self._reconcile_thread.start()

    def _check_reconcile(self):
        with self._reconcile_lock:
            result = self._reconcile_result
            self._reconcile_result = None
        if result is None:
            return
        self._reconcile_thread = None
        self.world.apply_update(result)
        if result.person_updates:
            self.people.apply_updates(result.person_updates)

    # --- Synchronous microservices ---

    def _run_script_check(self):
        """Run script_check + thought synchronously, handle advance/off_book."""
        beat = self.script.current_beat if self.script else None
        if beat is None or not self._beats_enabled():
            return

        try:
            check = script_check_call(
                beat=beat,
                history=self.session.get_history(),
                model_config=self.micro_config,
                world=self.world,
            )
        except Exception as e:
            self.errors.append(f"script_check: {e}")
            return

        # Thought
        if self._thinker_enabled():
            try:
                tresult = thought_call(
                    system_prompt=self.session.system_prompt,
                    history=self.session.get_history(),
                    model_config=self.micro_config,
                    world=self.world,
                    goals=self.goals,
                )
                if tresult.thought:
                    self.thoughts_generated += 1
                    self.session.inject_system(f"[Inner thought: {tresult.thought}]")
            except Exception:
                pass

        if check.status == "advance":
            self.script.advance()
            if not self.script.current_beat:
                self._replan("")
        elif check.status == "off_book" and check.plan_request:
            self._replan(check.plan_request)

    def _run_director(self):
        """Run director synchronously."""
        if not self._director_enabled() or self.scenario is None:
            return
        try:
            result = director_call(
                scenario=self.scenario,
                world=self.world,
                people=self.people,
                history=self.session.get_history(),
                model_config=self.micro_config,
            )
        except Exception as e:
            self.errors.append(f"director: {e}")
            return

        if result.status == "advance" and result.exit_index >= 0:
            stage = self.scenario.active_stage
            if stage and 0 <= result.exit_index < len(stage.exits):
                target = stage.exits[result.exit_index].goto
                if self.scenario.advance_to(target):
                    self.stages_advanced += 1
                    self.log.append({"type": "director", "status": "advance", "target": target})
                    self._replan("")

    def _replan(self, plan_request: str = ""):
        """Synchronous single-beat replan."""
        if not self._beats_enabled():
            return
        stage_goal = self.scenario.active_stage.goal if self.scenario and self.scenario.active_stage else ""
        try:
            result = next_beat_call(
                system_prompt=self.session.system_prompt,
                world=self.world,
                history=self.session.get_history(),
                goals=self.goals,
                model_config=self.micro_config,
                plan_request=plan_request,
                people=self.people,
                stage_goal=stage_goal,
            )
            if result.beats:
                self.script.replace(result.beats)
                self.beats_generated += len(result.beats)
        except Exception as e:
            self.errors.append(f"replan: {e}")

    # --- Public API ---

    def boot(self):
        """Synchronous initial plan."""
        if not self._beats_enabled():
            return
        stage_goal = self.scenario.active_stage.goal if self.scenario and self.scenario.active_stage else ""
        try:
            result = next_beat_call(
                system_prompt=self.session.system_prompt,
                world=self.world,
                history=self.session.get_history(),
                goals=self.goals,
                model_config=self.micro_config,
                people=self.people,
                stage_goal=stage_goal,
            )
            if result.beats:
                self.script.replace(result.beats)
                self.beats_generated += len(result.beats)
        except Exception as e:
            self.errors.append(f"boot: {e}")

    def send_message(self, text: str) -> str:
        self._check_reconcile()

        if not self._had_user_input:
            self._had_user_input = True
            response = self._collect_response(text)
            self.log.append({"type": "send", "input": text, "response": response})
            self._run_script_check()
            self._run_director()
            self._replan("")
            return response

        # Bootstrap
        if self.script.is_empty() and self._beats_enabled():
            self._replan("")

        if self.script.current_beat is not None:
            beat = self.script.current_beat
            guidance = load_beat_guide(beat.intent, beat.line)
            self.session.inject_system(guidance)
            response = self._collect_response(text)
        else:
            response = self._collect_response(text)

        self.log.append({"type": "send", "input": text, "response": response})
        self._run_script_check()
        self._run_director()
        return response

    def send_world(self, text: str) -> str:
        self._check_reconcile()
        self.world.add_pending(text)
        narrator_msg = format_pending_narrator(text)
        self.session.inject_system(narrator_msg)
        response = self._collect_response("[React to what just happened.]")
        self.log.append({"type": "world", "input": text, "response": response})
        self._start_reconcile()
        self._run_script_check()
        self._run_director()
        return response

    def send_beat(self) -> str:
        self._check_reconcile()

        if self.script.current_beat is None:
            if self._beats_enabled():
                self._replan("")
            if self.script.current_beat is None:
                if not self._beats_enabled():
                    # Improvise
                    response = self._collect_response("[Continue the conversation naturally.]")
                    self.log.append({"type": "beat", "response": response})
                    return response
                self.log.append({"type": "beat", "response": ""})
                return ""

        beat = self.script.current_beat

        # Condition check
        if beat.condition and self._beats_enabled():
            try:
                cond_result = condition_check_call(
                    condition=beat.condition,
                    system_prompt=self.session.system_prompt,
                    world=self.world,
                    history=self.session.get_history(),
                    model_config=self.micro_config,
                )
                if not cond_result.met:
                    idle = cond_result.idle or ""
                    if idle:
                        self.session.add_assistant(idle)
                    self.log.append({"type": "beat", "response": idle, "condition_met": False})
                    return idle
            except Exception:
                pass

        # Deliver beat verbatim
        response = beat.line
        self.session.add_assistant(response)
        self.script.advance()
        self.beats_generated += 1

        if not self.script.current_beat and self._beats_enabled():
            self._replan("")

        self._run_script_check()
        self._run_director()
        self.log.append({"type": "beat", "response": response, "intent": beat.intent})
        return response

    def send_trigger(self, trigger_num: int) -> str:
        """Simulate a stage trigger (1-based)."""
        self._check_reconcile()
        if self.scenario is None or self.scenario.active_stage is None:
            return ""

        stage = self.scenario.active_stage
        exit_index = trigger_num - 1
        if exit_index < 0 or exit_index >= len(stage.exits):
            return ""

        exit_obj = stage.exits[exit_index]
        old_stage = stage.name
        if not self.scenario.advance_to(exit_obj.goto):
            return ""

        self.stages_advanced += 1
        new_stage = self.scenario.active_stage

        # Inject exit condition as perception event
        event = PerceptionEvent(description=exit_obj.condition, source="trigger")
        _, narrator_msg = process_perception(event, self.people, self.world)
        self.session.inject_system(narrator_msg)
        response = self._collect_response("[React to what you just noticed.]")

        self.log.append({
            "type": "trigger",
            "trigger": trigger_num,
            "old_stage": old_stage,
            "new_stage": new_stage.name if new_stage else None,
            "response": response,
        })

        # Replan for new stage
        self.script.replace([])
        if self._beats_enabled():
            self._replan("")

        self._start_reconcile()
        self._run_director()
        return response

    def wait_for_background(self, timeout: float = 10.0):
        if self._reconcile_thread is not None and self._reconcile_thread.is_alive():
            self._reconcile_thread.join(timeout=timeout)
        self._check_reconcile()


# ---------------------------------------------------------------------------
# Per-turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    action: str
    input_text: str
    response: str
    latency_ms: float
    beat_intent: str = ""
    stage_name: str = ""
    error: str = ""


@dataclass
class ComboResult:
    combo: str
    model: str
    model_name: str
    description: str
    toggles: dict
    micro_model: str = ""
    micro_model_name: str = ""
    turns: list[TurnResult] = field(default_factory=list)
    stages_advanced: int = 0
    beats_generated: int = 0
    thoughts_generated: int = 0
    total_latency_ms: float = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_combo(model_key: str, model_config: dict, combo: ToggleCombo,
              micro_model_config: dict | None = None) -> ComboResult:
    """Run the scripted conversation with one model × one toggle combo."""
    micro_cfg = micro_model_config or model_config
    micro_key = ""
    micro_name = ""
    if micro_model_config:
        # Find the key for this config
        for k, v in MODELS.items():
            if v is micro_model_config:
                micro_key = k
                break
        micro_name = micro_model_config.get("name", "")

    result = ComboResult(
        combo=combo.name,
        model=model_key,
        model_name=model_config["name"],
        description=combo.description,
        toggles={"beats": combo.beats, "thinker": combo.thinker, "director": combo.director},
        micro_model=micro_key,
        micro_model_name=micro_name,
    )

    try:
        driver = ToggleDriver(CHARACTER, model_config, combo, micro_model_config=micro_model_config)
        driver.boot()

        for i, (action, text) in enumerate(SCRIPTED_TURNS):
            stage_name = ""
            if driver.scenario and driver.scenario.active_stage:
                stage_name = driver.scenario.active_stage.name

            t0 = time.perf_counter()
            beat_intent = ""
            error = ""

            try:
                if action == "send":
                    response = driver.send_message(text)
                elif action == "world":
                    response = driver.send_world(text)
                elif action == "beat":
                    if driver.script.current_beat:
                        beat_intent = driver.script.current_beat.intent
                    response = driver.send_beat()
                elif action == "trigger":
                    response = driver.send_trigger(int(text))
                else:
                    response = ""
            except Exception as e:
                response = ""
                error = str(e)
                driver.errors.append(f"turn {i}: {e}")

            latency = (time.perf_counter() - t0) * 1000

            result.turns.append(TurnResult(
                turn=i,
                action=action,
                input_text=text,
                response=response,
                latency_ms=latency,
                beat_intent=beat_intent,
                stage_name=stage_name,
                error=error,
            ))

        driver.wait_for_background()
        result.stages_advanced = driver.stages_advanced
        result.beats_generated = driver.beats_generated
        result.thoughts_generated = driver.thoughts_generated
        result.total_latency_ms = sum(t.latency_ms for t in result.turns)
        result.errors = driver.errors

    except Exception as e:
        result.errors.append(f"driver: {e}")

    return result


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------


def save_log(result: ComboResult) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"qa_toggles_{result.model}_{result.combo}_{timestamp}.json"
    data = {
        "type": "qa_toggles",
        "model": result.model,
        "model_name": result.model_name,
        "micro_model": result.micro_model,
        "micro_model_name": result.micro_model_name,
        "combo": result.combo,
        "description": result.description,
        "toggles": result.toggles,
        "timestamp": datetime.now().isoformat(),
        "stages_advanced": result.stages_advanced,
        "beats_generated": result.beats_generated,
        "thoughts_generated": result.thoughts_generated,
        "total_latency_ms": result.total_latency_ms,
        "errors": result.errors,
        "turns": [
            {
                "turn": t.turn,
                "action": t.action,
                "input_text": t.input_text,
                "response": t.response,
                "latency_ms": t.latency_ms,
                "beat_intent": t.beat_intent,
                "stage_name": t.stage_name,
                "error": t.error,
            }
            for t in result.turns
        ],
    }
    log_path.write_text(json.dumps(data, indent=2))
    return log_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(results: list[ComboResult]):
    table = Table(title="Toggle Test Results")
    table.add_column("Combo", style="bold")
    table.add_column("Model")
    table.add_column("Stages", justify="right")
    table.add_column("Beats", justify="right")
    table.add_column("Thoughts", justify="right")
    table.add_column("Total ms", justify="right")
    table.add_column("Errors", justify="right")

    for r in results:
        err_style = "[red]" if r.errors else ""
        err_end = "[/red]" if r.errors else ""
        table.add_row(
            r.combo, r.model_name,
            str(r.stages_advanced), str(r.beats_generated), str(r.thoughts_generated),
            f"{r.total_latency_ms:.0f}",
            f"{err_style}{len(r.errors)}{err_end}",
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# HTML merge report
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def merge_and_report(paths: list[str]) -> None:
    logs: list[dict] = []
    for p in paths:
        logs.append(json.loads(Path(p).read_text()))

    # Group by (combo, model)
    by_combo: dict[str, dict[str, dict]] = {}
    models_seen: list[str] = []
    model_names: dict[str, str] = {}
    combo_descriptions: dict[str, str] = {}
    combo_toggles: dict[str, dict] = {}

    for log in logs:
        combo = log["combo"]
        model_key = log["model"]
        if model_key not in models_seen:
            models_seen.append(model_key)
        model_names[model_key] = log["model_name"]
        combo_descriptions[combo] = log.get("description", "")
        combo_toggles[combo] = log.get("toggles", {})
        if combo not in by_combo:
            by_combo[combo] = {}
        by_combo[combo][model_key] = log

    # Ordered combos
    combo_order = [c.name for c in TOGGLE_COMBOS]
    combos = [c for c in combo_order if c in by_combo]
    # Add any extras not in the predefined order
    for c in by_combo:
        if c not in combos:
            combos.append(c)

    # Build matrix table rows
    matrix_html = ""
    for combo in combos:
        model_data = by_combo[combo]
        desc = _esc(combo_descriptions.get(combo, ""))
        toggles = combo_toggles.get(combo, {})
        toggle_badges = ""
        for tname, tval in toggles.items():
            color = "#3fb950" if tval else "#f85149"
            toggle_badges += f'<span style="color:{color};font-size:0.8em;margin-right:0.5em">{_esc(tname)}:{"on" if tval else "off"}</span>'

        matrix_html += f'<tr><td class="combo-cell"><b>{_esc(combo)}</b><br><span class="desc">{desc}</span><br>{toggle_badges}</td>'
        for m in models_seen:
            d = model_data.get(m)
            if d is None:
                matrix_html += '<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
                continue
            errs = len(d.get("errors", []))
            err_html = f'<span class="fail">{errs}</span>' if errs else "0"
            matrix_html += (
                f'<td class="num">{d.get("stages_advanced", 0)}</td>'
                f'<td class="num">{d.get("beats_generated", 0)}</td>'
                f'<td class="num">{d.get("thoughts_generated", 0)}</td>'
                f'<td class="num">{d.get("total_latency_ms", 0):.0f}</td>'
                f'<td class="num">{err_html}</td>'
            )
        matrix_html += '</tr>\n'

    # Model header columns
    model_cols = ""
    sub_cols = ""
    for m in models_seen:
        mname = model_names.get(m, m)
        model_cols += f'<th colspan="5">{_esc(mname)}</th>'
        sub_cols += '<th>Stages</th><th>Beats</th><th>Thoughts</th><th>ms</th><th>Errs</th>'

    # Build per-combo turn-by-turn side-by-side
    turns_html = ""
    for combo in combos:
        model_data = by_combo[combo]
        desc = _esc(combo_descriptions.get(combo, ""))
        turns_html += f'<div class="comparison"><h3>{_esc(combo)}</h3><p class="desc">{desc}</p>'
        turns_html += '<div class="side-by-side">'

        for m in models_seen:
            d = model_data.get(m)
            mname = model_names.get(m, m)
            turns_html += f'<div class="model-output"><h4>{_esc(mname)}</h4>'
            if d is None:
                turns_html += '<p>(no data)</p>'
            else:
                for t in d.get("turns", []):
                    action = t.get("action", "")
                    input_text = t.get("input_text", "")
                    response = t.get("response", "")
                    latency = t.get("latency_ms", 0)
                    stage = t.get("stage_name", "")
                    beat_intent = t.get("beat_intent", "")
                    error = t.get("error", "")

                    action_color = {
                        "send": "#58a6ff", "world": "#d29922", "beat": "#bc8cff",
                        "trigger": "#f0883e",
                    }.get(action, "#8b949e")

                    stage_tag = f' <span class="stage-tag">{_esc(stage)}</span>' if stage else ""
                    intent_tag = f' <span class="intent-tag">[{_esc(beat_intent)}]</span>' if beat_intent else ""
                    err_tag = f' <span class="fail">{_esc(error)}</span>' if error else ""

                    turns_html += (
                        f'<div class="turn-row">'
                        f'<div class="turn-meta">'
                        f'<span class="badge" style="background:{action_color}">{_esc(action)}</span>'
                        f' <span class="latency">{latency:.0f}ms</span>'
                        f'{stage_tag}{intent_tag}{err_tag}'
                        f'</div>'
                    )
                    if input_text:
                        turns_html += f'<div class="input">{_esc(input_text)}</div>'
                    if response:
                        turns_html += f'<div class="response">{_esc(response)}</div>'
                    turns_html += '</div>'

                if d.get("errors"):
                    turns_html += '<div class="error-list"><b>Errors:</b><ul>'
                    for e in d["errors"]:
                        turns_html += f'<li>{_esc(e)}</li>'
                    turns_html += '</ul></div>'

            turns_html += '</div>'
        turns_html += '</div></div>\n'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QA Toggles Comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1600px; margin: 2em auto; padding: 0 1em; background: #0d1117; color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.combo-cell {{ min-width: 200px; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .desc {{ color: #8b949e; font-size: 0.85em; font-style: italic; }}
  .pass {{ color: #3fb950; font-weight: bold; }}
  .fail {{ color: #f85149; font-weight: bold; }}
  .comparison {{ margin: 2em 0; border: 1px solid #30363d; border-radius: 6px; padding: 1em; }}
  .side-by-side {{ display: flex; gap: 1em; }}
  .model-output {{ flex: 1; min-width: 0; }}
  .model-output h4 {{ color: #58a6ff; margin: 0 0 0.5em; }}
  .turn-row {{ padding: 0.4em 0; border-bottom: 1px solid #21262d; }}
  .turn-row:last-child {{ border-bottom: none; }}
  .turn-meta {{ font-size: 0.85em; color: #8b949e; margin-bottom: 0.2em; }}
  .badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.8em; color: #fff; font-weight: 600; }}
  .latency {{ color: #8b949e; font-size: 0.85em; }}
  .stage-tag {{ color: #f0883e; font-size: 0.85em; }}
  .intent-tag {{ color: #bc8cff; font-size: 0.85em; }}
  .input {{ color: #58a6ff; margin: 0.2em 0; padding-left: 1em; font-size: 0.9em; }}
  .response {{ color: #7ee787; margin: 0.2em 0; padding-left: 1em; border-left: 3px solid #238636; padding: 0.3em 0.8em; background: #0d1117; border-radius: 0 4px 4px 0; font-size: 0.9em; white-space: pre-wrap; word-break: break-word; }}
  .error-list {{ color: #f85149; font-size: 0.85em; margin-top: 0.5em; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
</style></head><body>
<h1>QA Toggles Comparison</h1>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; Models: {', '.join(_esc(model_names.get(m, m)) for m in models_seen)}</p>

<h2>Summary Matrix</h2>
<table>
<tr><th>Combo</th>{model_cols}</tr>
<tr><th></th>{sub_cols}</tr>
{matrix_html}</table>

<h2>Turn-by-Turn Comparison</h2>
{turns_html}
</body></html>"""

    LOGS_DIR.mkdir(exist_ok=True)
    html_path = LOGS_DIR / f"qa_toggles_comparison_{timestamp}.html"
    html_path.write_text(html)
    console.print(f"[bold]Comparison report: {html_path}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Scaffold toggle testing")
    parser.add_argument("--model", default=None, choices=list(MODELS.keys()),
                        help="Single model to test all combos")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model keys (e.g. groq-llama-8b,kimi-k2)")
    parser.add_argument("--micro-model", default=None, choices=list(MODELS.keys()),
                        help="Model for microservice calls (script_check, thought, director, beat, reconcile)")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="Merge existing toggle QA logs into comparison HTML")
    args = parser.parse_args()

    if args.merge:
        merge_and_report(args.merge)
        return

    # Determine models to test
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in MODELS:
                console.print(f"[red]Unknown model: {k}[/red]")
                sys.exit(1)
    elif args.model:
        model_keys = [args.model]
    else:
        model_keys = [DEFAULT_MODEL]

    # Resolve micro-model config
    micro_model_config = None
    if args.micro_model:
        micro_model_config = MODELS[args.micro_model]

    all_results: list[ComboResult] = []
    log_paths: list[Path] = []

    for model_key in model_keys:
        model_config = MODELS[model_key]
        micro_label = f", micro: {micro_model_config['name']}" if micro_model_config else ""
        console.print(f"\n[bold]Model: {model_config['name']}{micro_label}[/bold]")

        for combo in TOGGLE_COMBOS:
            console.print(f"  [dim]{combo.name}[/dim] — {combo.description} ...", end=" ")
            result = run_combo(model_key, model_config, combo, micro_model_config=micro_model_config)
            all_results.append(result)

            status = "[green]OK[/green]" if not result.errors else f"[red]{len(result.errors)} errors[/red]"
            console.print(f"{status} ({result.total_latency_ms:.0f}ms, {result.stages_advanced} stages, {result.beats_generated} beats)")

            path = save_log(result)
            log_paths.append(path)
            console.print(f"    [dim]{path}[/dim]")

    print_summary(all_results)

    # Auto-generate comparison if multiple runs
    if len(all_results) > 1:
        console.print(f"\n[dim]Generating comparison report...[/dim]")
        merge_and_report([str(p) for p in log_paths])


if __name__ == "__main__":
    main()
