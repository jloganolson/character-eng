"""Persona-based subjective QA testing.

Launches 6 LLM-driven personas that chat with a character in parallel,
exercising the full async loop (background eval/plan/reconcile). Produces
an HTML report for human review.

Usage: uv run -m character_eng.qa_personas [--model cerebras-llama] [--character greg] [--turns 10]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from character_eng.chat import ChatSession
from character_eng.models import BIG_MODEL, DEFAULT_MODEL, MODELS
from character_eng.perception import PerceptionEvent, process_perception
from character_eng.person import PeopleState
from character_eng.prompts import load_prompt
from character_eng.scenario import DirectorResult, director_call, load_scenario_script
from character_eng.world import (
    Beat,
    ConditionResult,
    EvalResult,
    Goals,
    PlanResult,
    Script,
    WorldUpdate,
    condition_check_call,
    eval_call,
    format_pending_narrator,
    load_beat_guide,
    load_goals,
    load_world_state,
    plan_call,
    reconcile_call,
)

load_dotenv()

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CHARACTER = "greg"

# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------


def _get_model_config(model_key: str) -> dict | None:
    """Return model config if the key exists and its API key is set."""
    cfg = MODELS.get(model_key)
    if not cfg:
        return None
    env = cfg.get("api_key_env", "")
    if env and not os.environ.get(env):
        return None
    return cfg


def _get_big_model_config() -> dict | None:
    return _get_model_config(BIG_MODEL)


# ---------------------------------------------------------------------------
# ConversationDriver — full __main__.py loop as instance state
# ---------------------------------------------------------------------------


class ConversationDriver:
    """Encapsulates the full background threading loop so multiple personas
    can run in parallel without interference."""

    def __init__(self, character: str, model_config: dict):
        self.character = character
        self.model_config = model_config
        big_cfg = _get_big_model_config()
        self.big_model_config = big_cfg
        self.eval_model_config = big_cfg if big_cfg else model_config

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

        # Background eval
        self._eval_lock = threading.Lock()
        self._eval_result: EvalResult | None = None
        self._eval_thread: threading.Thread | None = None
        self._eval_version: int = 0

        # Background plan
        self._plan_lock = threading.Lock()
        self._plan_result: PlanResult | None = None
        self._plan_thread: threading.Thread | None = None
        self._plan_version: int = 0

        # Background director
        self._director_lock = threading.Lock()
        self._director_result: DirectorResult | None = None
        self._director_thread: threading.Thread | None = None
        self._director_version: int = 0

        # Context version counter
        self._context_version: int = 0

        # Conversation log
        self.log: list[dict] = []
        self._had_user_input = False

    # --- Version tracking ---

    def _bump_version(self):
        self._context_version += 1

    # --- Collect response (no streaming display) ---

    def _collect_response(self, message: str) -> str:
        """Consume session.send() generator, return full text."""
        chunks = []
        for chunk in self.session.send(message):
            chunks.append(chunk)
        return "".join(chunks)

    # --- Background reconcile ---

    def _run_reconcile(self, pending: list[str], cfg: dict):
        try:
            result = reconcile_call(self.world, pending, cfg, people=self.people)
            with self._reconcile_lock:
                self._reconcile_result = result
        except Exception:
            pass  # silently discard in QA context

    def _start_reconcile(self):
        pending = self.world.clear_pending()
        if not pending:
            return
        cfg = self.big_model_config if self.big_model_config else self.model_config
        self._reconcile_thread = threading.Thread(
            target=self._run_reconcile, args=(pending, cfg), daemon=True
        )
        self._reconcile_thread.start()

    def _check_reconcile(self) -> dict | None:
        """Check if background reconcile is ready. Returns log entry or None."""
        with self._reconcile_lock:
            result = self._reconcile_result
            self._reconcile_result = None
        if result is None:
            return None
        self._reconcile_thread = None
        self.world.apply_update(result)
        if result.person_updates:
            self.people.apply_updates(result.person_updates)
        return {
            "type": "reconcile",
            "remove_facts": result.remove_facts,
            "add_facts": result.add_facts,
            "events": result.events,
        }

    # --- Background eval ---

    def _run_eval(self, system_prompt, history, goals, script, model_config, people, stage_goal):
        try:
            result = eval_call(
                system_prompt=system_prompt,
                world=self.world,
                history=history,
                model_config=model_config,
                goals=goals,
                script=script,
                people=people,
                stage_goal=stage_goal,
            )
            with self._eval_lock:
                self._eval_result = result
        except Exception:
            pass

    def _start_eval(self):
        if self._eval_thread is not None and self._eval_thread.is_alive():
            return
        self._eval_version = self._context_version
        stage_goal = ""
        if self.scenario and self.scenario.active_stage:
            stage_goal = self.scenario.active_stage.goal
        self._eval_thread = threading.Thread(
            target=self._run_eval,
            args=(
                self.session.system_prompt,
                self.session.get_history(),
                self.goals,
                self.script,
                self.eval_model_config,
                self.people,
                stage_goal,
            ),
            daemon=True,
        )
        self._eval_thread.start()

    def _check_eval(self) -> tuple[bool, str, dict | None]:
        """Returns (needs_plan, plan_request, eval_log_entry)."""
        with self._eval_lock:
            result = self._eval_result
            self._eval_result = None
        if result is None:
            return (False, "", None)
        self._eval_thread = None

        if self._eval_version != self._context_version:
            return (False, "", {"type": "eval_stale"})

        # Inject eval into session
        parts = [f"[Inner thought: {result.thought}]"]
        if result.gaze:
            parts.append(f"[gaze:{result.gaze}]")
        if result.expression:
            parts.append(f"[emote:{result.expression}]")
        self.session.inject_system(" ".join(parts))

        entry = {
            "type": "eval",
            "thought": result.thought,
            "gaze": result.gaze,
            "expression": result.expression,
            "script_status": result.script_status,
        }
        if result.plan_request:
            entry["plan_request"] = result.plan_request

        if result.script_status == "advance":
            self.script.advance()
            if not self.script.current_beat:
                return (True, "", entry)
            return (False, "", entry)
        elif result.script_status == "off_book" and result.plan_request:
            return (True, result.plan_request, entry)

        return (False, "", entry)

    # --- Background plan ---

    def _run_plan_bg(self, system_prompt, history, goals, plan_request, big_model_config, people, stage_goal):
        try:
            result = plan_call(
                system_prompt=system_prompt,
                world=self.world,
                history=history,
                goals=goals,
                plan_request=plan_request,
                plan_model_config=big_model_config,
                people=people,
                stage_goal=stage_goal,
            )
            with self._plan_lock:
                self._plan_result = result
        except Exception:
            pass

    def _start_plan(self, plan_request: str = ""):
        if self.big_model_config is None:
            return
        if self._plan_thread is not None and self._plan_thread.is_alive():
            return
        self._plan_version = self._context_version
        stage_goal = ""
        if self.scenario and self.scenario.active_stage:
            stage_goal = self.scenario.active_stage.goal
        self._plan_thread = threading.Thread(
            target=self._run_plan_bg,
            args=(
                self.session.system_prompt,
                self.session.get_history(),
                self.goals,
                plan_request,
                self.big_model_config,
                self.people,
                stage_goal,
            ),
            daemon=True,
        )
        self._plan_thread.start()

    def _check_plan(self) -> dict | None:
        """Returns log entry or None."""
        with self._plan_lock:
            result = self._plan_result
            self._plan_result = None
        if result is None:
            return None
        self._plan_thread = None

        if self._plan_version != self._context_version:
            return {"type": "plan_stale"}

        if result.beats:
            self.script.replace(result.beats)
            return {
                "type": "plan",
                "beats": len(result.beats),
                "intents": [b.intent for b in result.beats],
            }
        return None

    # --- Background director ---

    def _run_director(self, scenario, world, people, history, model_config):
        try:
            result = director_call(
                scenario=scenario,
                world=world,
                people=people,
                history=history,
                model_config=model_config,
            )
            with self._director_lock:
                self._director_result = result
        except Exception:
            pass

    def _start_director(self):
        if self.scenario is None:
            return
        if self._director_thread is not None and self._director_thread.is_alive():
            return
        self._director_version = self._context_version
        cfg = self.big_model_config if self.big_model_config else self.model_config
        self._director_thread = threading.Thread(
            target=self._run_director,
            args=(
                self.scenario,
                self.world,
                self.people,
                self.session.get_history(),
                cfg,
            ),
            daemon=True,
        )
        self._director_thread.start()

    def _check_director(self) -> dict | None:
        """Check if background director is ready. Returns log entry or None."""
        with self._director_lock:
            result = self._director_result
            self._director_result = None
        if result is None:
            return None
        self._director_thread = None

        if self._director_version != self._context_version:
            return {"type": "director_stale"}

        entry = {
            "type": "director",
            "thought": result.thought,
            "status": result.status,
            "exit_index": result.exit_index,
        }

        if result.status == "advance" and self.scenario:
            stage = self.scenario.active_stage
            if stage and 0 <= result.exit_index < len(stage.exits):
                target = stage.exits[result.exit_index].goto
                self.scenario.advance_to(target)
                entry["advanced_to"] = target
                # Trigger replan with new stage goal
                self._start_plan("")

        return entry

    # --- Turn-start checks ---

    def _turn_start_checks(self):
        """Run all turn-start background result checks. Appends to log."""
        if self.world is not None:
            entry = self._check_reconcile()
            if entry:
                self.log.append(entry)

        needs_plan, plan_request, eval_entry = self._check_eval()
        if eval_entry:
            self.log.append(eval_entry)
        if needs_plan:
            self._start_plan(plan_request)

        director_entry = self._check_director()
        if director_entry:
            self.log.append(director_entry)

        plan_entry = self._check_plan()
        if plan_entry:
            self.log.append(plan_entry)

    # --- Public API ---

    def boot(self):
        """Synchronous initial plan (same as __main__.py boot)."""
        if self.big_model_config is None:
            return
        stage_goal = ""
        if self.scenario and self.scenario.active_stage:
            stage_goal = self.scenario.active_stage.goal
        try:
            result = plan_call(
                system_prompt=self.session.system_prompt,
                world=self.world,
                history=self.session.get_history(),
                goals=self.goals,
                plan_request="",
                plan_model_config=self.big_model_config,
                people=self.people,
                stage_goal=stage_goal,
            )
            if result and result.beats:
                self.script.replace(result.beats)
                self.log.append({
                    "type": "plan",
                    "beats": len(result.beats),
                    "intents": [b.intent for b in result.beats],
                })
        except Exception as e:
            self.log.append({"type": "plan_error", "error": str(e)})

    def send_message(self, text: str) -> str:
        """Full turn: turn-start checks, guided/unguided beat, background eval.
        Returns character response text."""
        self._turn_start_checks()

        # First user input: natural LLM response + background eval + replan
        if not self._had_user_input:
            self._had_user_input = True
            response = self._collect_response(text)
            self.log.append({"type": "send", "input": text, "response": response})
            self._bump_version()
            self._start_eval()
            self._start_director()
            self._start_plan("")
            return response

        # Bootstrap: if no script, start background plan
        if self.script.is_empty():
            self._start_plan("")

        if self.script.current_beat is not None:
            # LLM-guided beat delivery
            beat = self.script.current_beat
            guidance = load_beat_guide(beat.intent, beat.line)
            self.session.inject_system(guidance)
            response = self._collect_response(text)
            # Inject beat metadata
            meta_parts = []
            if beat.gaze:
                meta_parts.append(f"[gaze:{beat.gaze}]")
            if beat.expression:
                meta_parts.append(f"[emote:{beat.expression}]")
            if meta_parts:
                self.session.inject_system(" ".join(meta_parts))
        else:
            # No script — LLM generates response
            response = self._collect_response(text)

        self.log.append({"type": "send", "input": text, "response": response})
        self._bump_version()
        self._start_eval()
        self._start_director()
        return response

    def send_world(self, text: str) -> str:
        """World change: narrator injection, character reaction, background reconcile + eval.
        Returns character response text."""
        self._turn_start_checks()

        if self.world is None:
            return ""

        self.world.add_pending(text)
        narrator_msg = format_pending_narrator(text)
        self.session.inject_system(narrator_msg)
        response = self._collect_response("[React to what just happened.]")

        self.log.append({
            "type": "world",
            "input": text,
            "narrator": narrator_msg,
            "response": response,
        })

        self._start_reconcile()
        self._bump_version()
        self._start_eval()
        self._start_director()
        return response

    def send_beat(self) -> str:
        """Deliver next scripted beat. Condition check sync, replan if no script.
        Returns character response text or idle line."""
        self._turn_start_checks()

        entry: dict = {"type": "beat"}

        # If no current beat, synchronous replan
        if self.script.current_beat is None:
            if self.big_model_config:
                stage_goal = ""
                if self.scenario and self.scenario.active_stage:
                    stage_goal = self.scenario.active_stage.goal
                try:
                    result = plan_call(
                        system_prompt=self.session.system_prompt,
                        world=self.world,
                        history=self.session.get_history(),
                        goals=self.goals,
                        plan_request="",
                        plan_model_config=self.big_model_config,
                        people=self.people,
                        stage_goal=stage_goal,
                    )
                    if result and result.beats:
                        self.script.replace(result.beats)
                except Exception:
                    pass
            if self.script.current_beat is None:
                entry["response"] = ""
                self.log.append(entry)
                return ""

        beat = self.script.current_beat

        # Condition check
        if beat.condition:
            try:
                cond_result = condition_check_call(
                    condition=beat.condition,
                    system_prompt=self.session.system_prompt,
                    world=self.world,
                    history=self.session.get_history(),
                    model_config=self.model_config,
                )
                entry["condition"] = beat.condition
                entry["condition_met"] = cond_result.met
                if not cond_result.met:
                    idle = cond_result.idle or ""
                    if idle:
                        self.session.add_assistant(idle)
                    entry["response"] = idle
                    self.log.append(entry)
                    return idle
            except Exception:
                pass

        # Deliver beat verbatim
        response = beat.line
        self.session.add_assistant(response)

        # Inject metadata
        meta_parts = []
        if beat.gaze:
            meta_parts.append(f"[gaze:{beat.gaze}]")
        if beat.expression:
            meta_parts.append(f"[emote:{beat.expression}]")
        if meta_parts:
            self.session.inject_system(" ".join(meta_parts))

        entry["response"] = response
        entry["intent"] = beat.intent
        if beat.gaze:
            entry["gaze"] = beat.gaze
        if beat.expression:
            entry["expression"] = beat.expression

        # Advance
        self.script.advance()
        if not self.script.current_beat:
            self._start_plan("")

        self._bump_version()
        self._start_eval()
        self._start_director()
        self.log.append(entry)
        return response

    def send_see(self, text: str) -> str:
        """Perception event: process, inject narrator, collect response, start background.
        Returns character response text."""
        self._turn_start_checks()

        event = PerceptionEvent(description=text, source="persona")
        _, narrator_msg = process_perception(event, self.people, self.world)
        self.session.inject_system(narrator_msg)
        response = self._collect_response("[React to what you just noticed.]")

        self.log.append({
            "type": "see",
            "input": text,
            "narrator": narrator_msg,
            "response": response,
        })

        self._start_reconcile()
        self._bump_version()
        self._start_eval()
        self._start_director()
        return response

    def wait_for_background(self, timeout: float = 15.0):
        """Join any active background threads."""
        for t in [self._reconcile_thread, self._eval_thread, self._plan_thread, self._director_thread]:
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
        # Final drain of results
        self._turn_start_checks()


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONA_SYSTEM = """You are simulating a player interacting with an NPC in a video game.
You MUST respond with valid JSON only — no other text.

Output format:
{"action": "<action>", "text": "<text>"}

action must be one of: "message", "world", "beat", "see", "nothing"
- "message": send a chat message (put your message in "text")
- "world": describe a world change (put description in "text")
- "beat": request the next scripted beat (text is ignored)
- "see": describe something the character perceives (put description in "text")
- "nothing": do nothing this turn (text is ignored)

Your persona:
"""


@dataclass
class PersonaConfig:
    name: str
    description: str
    turns: int
    persona_prompt: str  # appended to PERSONA_SYSTEM for LLM personas
    action_override: object = None  # Callable[[int], str] | None — forces action type per turn


PERSONAS: list[PersonaConfig] = [
    PersonaConfig(
        name="AFK Andy",
        description="Inaction griefer — minimal engagement",
        turns=12,
        persona_prompt=(
            "You are AFK Andy, a disengaged player who barely participates. "
            "You give minimal responses like 'k', 'idk', 'sure', 'ok', 'lol', 'whatever'. "
            "About 25% of the time, choose action 'nothing' to simulate being away. "
            "Never use /world or /beat. Always action 'message' or 'nothing'. "
            "Keep messages very short (1-3 words)."
        ),
    ),
    PersonaConfig(
        name="Chaos Carl",
        description="Overaction griefer — fires, explosions, chaos",
        turns=12,
        persona_prompt=(
            "You are Chaos Carl, a chaotic player who loves causing mayhem. "
            "About 40% of the time, use action 'world' to describe wild changes "
            "(fires, explosions, alien invasions, the floor becoming lava, etc). "
            "The rest of the time, send bizarre messages — non sequiturs, "
            "demands to fight, random roleplay actions. Be creative and absurd. "
            "Never use 'beat' or 'nothing'."
        ),
    ),
    PersonaConfig(
        name="Casual Player",
        description="Normal user — natural engagement",
        turns=15,
        persona_prompt=(
            "You are a casual player having a genuine conversation with an NPC. "
            "Ask questions about the character, their world, and situation. "
            "Respond naturally to what they say. Show curiosity. "
            "Occasionally use 'world' (maybe 15% of turns) to describe gentle changes "
            "(someone walks by, weather changes, a sound is heard). "
            "Always action 'message' unless doing a world change. "
            "Write natural, conversational messages (1-2 sentences)."
        ),
    ),
    PersonaConfig(
        name="Bored Player",
        description="Not having fun — redirects and dismisses",
        turns=12,
        persona_prompt=(
            "You are a bored player who isn't enjoying this interaction. "
            "Complain about things being boring, try to redirect the conversation "
            "to unrelated topics (sports, food, memes), dismiss what the NPC says, "
            "and generally be difficult. Say things like 'this is boring', "
            "'can we talk about something else', 'I don't care about that'. "
            "Always action 'message'. Write short dismissive messages."
        ),
    ),
    PersonaConfig(
        name="Beat Runner",
        description="Scripted — every turn is /beat",
        turns=10,
        persona_prompt="",  # not used — action_override controls behavior
        action_override=lambda turn: "beat",
    ),
    PersonaConfig(
        name="Interrupter",
        description="Hybrid — alternates message and beat",
        turns=12,
        persona_prompt=(
            "You are a player who chats with the NPC between beats. "
            "Generate a short, natural message that responds to or comments on "
            "what the NPC just said. 1-2 sentences. Action is always 'message'."
        ),
        action_override=lambda turn: "message" if turn % 2 == 0 else "beat",
    ),
    PersonaConfig(
        name="Scene Observer",
        description="Perception tester — approaches, interacts, leaves",
        turns=15,
        persona_prompt=(
            "You are simulating a person approaching an NPC at a lemonade stand. "
            "You use a mix of 'see' and 'message' actions to simulate a scene. "
            "Use 'see' to describe things you or others do physically (approach, "
            "pick up items, gesture, look around, walk away). Use 'message' for "
            "dialogue. Early turns: approach and look around (see). Middle turns: "
            "chat and interact (mix of message and see). Final turns: say goodbye "
            "and leave (message then see). About 40% see, 60% message. "
            "Keep descriptions and dialogue short and natural."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Persona LLM — generates actions
# ---------------------------------------------------------------------------


def _make_persona_client(model_config: dict) -> OpenAI:
    api_key = model_config.get("api_key") or os.environ.get(model_config.get("api_key_env", ""), "")
    return OpenAI(api_key=api_key, base_url=model_config["base_url"])


def _get_persona_action(
    client: OpenAI,
    model_config: dict,
    persona: PersonaConfig,
    conversation_so_far: list[dict],
) -> dict:
    """Ask the persona LLM what to do next. Returns {"action": ..., "text": ...}."""
    messages = [
        {"role": "system", "content": PERSONA_SYSTEM + persona.persona_prompt},
    ]
    # Add conversation context so the persona can react to what happened
    for entry in conversation_so_far[-6:]:
        if entry.get("type") == "send":
            messages.append({"role": "user", "content": f"You said: {entry['input']}\nNPC replied: {entry['response']}"})
        elif entry.get("type") == "world":
            messages.append({"role": "user", "content": f"You changed the world: {entry['input']}\nNPC reacted: {entry['response']}"})
        elif entry.get("type") == "see":
            messages.append({"role": "user", "content": f"You observed: {entry['input']}\nNPC reacted: {entry['response']}"})
        elif entry.get("type") == "beat":
            messages.append({"role": "user", "content": f"Beat delivered: {entry.get('response', '(none)')}"})
    messages.append({"role": "user", "content": "What do you do next?"})

    try:
        response = client.chat.completions.create(
            model=model_config["model"],
            messages=messages,
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        action = data.get("action", "message")
        text = data.get("text", "")
        if action not in ("message", "world", "beat", "see", "nothing"):
            action = "message"
        return {"action": action, "text": text}
    except Exception:
        return {"action": "message", "text": "hey"}


# ---------------------------------------------------------------------------
# Run a single persona
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn: int
    action: str
    input_text: str
    response: str
    eval_entry: dict | None = None
    stale_discard: bool = False


@dataclass
class PersonaResult:
    persona: PersonaConfig
    turns: list[TurnRecord] = field(default_factory=list)
    boot_log: list[dict] = field(default_factory=list)
    error: str | None = None


def run_persona(
    persona: PersonaConfig,
    character: str,
    model_config: dict,
    persona_model_config: dict,
    turn_override: int | None = None,
) -> PersonaResult:
    """Run a single persona's full conversation. Thread-safe."""
    result = PersonaResult(persona=persona)
    num_turns = turn_override if turn_override else persona.turns

    try:
        driver = ConversationDriver(character, model_config)
        driver.boot()
        result.boot_log = [e for e in driver.log]

        # Persona LLM client (only needed for LLM-driven personas)
        persona_client = None
        if persona.persona_prompt:
            persona_client = _make_persona_client(persona_model_config)

        for turn_num in range(num_turns):
            # Determine action
            if persona.action_override:
                action = persona.action_override(turn_num)
                text = ""
                # For hybrid personas with LLM messages
                if action == "message" and persona_client:
                    persona_action = _get_persona_action(
                        persona_client, persona_model_config, persona, driver.log
                    )
                    text = persona_action.get("text", "hey")
                elif action == "message":
                    text = "hey"
            elif persona_client:
                persona_action = _get_persona_action(
                    persona_client, persona_model_config, persona, driver.log
                )
                action = persona_action["action"]
                text = persona_action.get("text", "")
            else:
                action = "message"
                text = "hey"

            # Execute action
            record = TurnRecord(turn=turn_num, action=action, input_text=text, response="")

            if action == "nothing":
                record.response = "(idle)"
            elif action == "beat":
                record.response = driver.send_beat()
            elif action == "world":
                record.response = driver.send_world(text or "something strange happens")
            elif action == "see":
                record.response = driver.send_see(text or "someone walks by")
            else:  # message
                record.response = driver.send_message(text or "hey")

            # Check for stale discards in the log since last turn
            for entry in driver.log:
                if entry.get("type") in ("eval_stale", "plan_stale"):
                    record.stale_discard = True
                    break

            # Capture latest eval entry
            for entry in reversed(driver.log):
                if entry.get("type") == "eval":
                    record.eval_entry = entry
                    break

            result.turns.append(record)
            time.sleep(0.3)

        driver.wait_for_background(timeout=15.0)

    except Exception as e:
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def annotation_assets(report_name: str, model_key: str, character: str) -> str:
    """Return shared CSS + JS + toolbar HTML for annotation UI.

    Injected before </body> in both persona QA and chat session HTML reports.
    """
    return f"""
<style>
  .annotate-btn {{
    background: none; border: 1px solid #30363d; cursor: pointer;
    font-size: 0.8em; color: #8b949e; padding: 2px 8px;
    border-radius: 4px; margin-top: 0.3em; transition: all 0.15s;
  }}
  .annotate-btn:hover {{ border-color: #58a6ff; color: #58a6ff; }}
  .annotate-btn.has-note {{ border-color: #58a6ff; color: #58a6ff; }}
  .annotation-area {{
    display: none; margin: 0.4em 0 0.2em 1em;
  }}
  .annotation-area.open {{ display: block; }}
  .annotation-area textarea {{
    width: 100%; min-height: 3em; padding: 0.4em;
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; font-family: inherit; font-size: 0.9em;
    resize: vertical;
  }}
  .annotation-area textarea:focus {{ border-color: #58a6ff; outline: none; }}
  #annotation-toolbar {{
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #161b22; border-top: 1px solid #30363d;
    padding: 0.6em 1.5em; display: flex; align-items: center;
    gap: 1em; z-index: 100; font-size: 0.9em;
  }}
  #annotation-toolbar .count {{ color: #8b949e; }}
  #annotation-toolbar .count b {{ color: #58a6ff; }}
  #export-btn {{
    background: #238636; color: #fff; border: none; padding: 0.4em 1.2em;
    border-radius: 6px; cursor: pointer; font-size: 0.9em; font-weight: 600;
  }}
  #export-btn:hover {{ background: #2ea043; }}
  #export-btn:disabled {{ background: #30363d; color: #8b949e; cursor: default; }}
  #done-btn {{
    background: #30363d; color: #c9d1d9; border: none; padding: 0.4em 1.2em;
    border-radius: 6px; cursor: pointer; font-size: 0.9em; font-weight: 600;
    display: none;
  }}
  #done-btn:hover {{ background: #484f58; }}
  .save-flash {{
    color: #7ee787; font-weight: 600; opacity: 0;
    transition: opacity 0.3s;
  }}
  .save-flash.show {{ opacity: 1; }}
  .save-path {{ color: #8b949e; font-size: 0.85em; margin-left: auto; }}
  .save-path code {{ color: #c9d1d9; }}
  body {{ padding-bottom: 3.5em; }}
</style>

<div id="annotation-toolbar">
  <span class="count"><b id="note-count">0</b> annotations</span>
  <button id="export-btn" disabled onclick="exportAnnotations()">Export annotations</button>
  <button id="done-btn" onclick="shutdownServer()">Done</button>
  <span class="save-flash" id="save-flash"></span>
  <span class="save-path">Save to: <code>logs/annotated/{report_name}.annotations.json</code></span>
</div>

<script>
const REPORT_NAME = {json_esc(report_name)};
const MODEL_KEY = {json_esc(model_key)};
const CHARACTER = {json_esc(character)};
const annotations = new Map();
let lastSavedPath = '';

function toggleAnnotation(persona, turn) {{
  const key = persona + ':' + turn;
  const area = document.getElementById('ann-' + css_safe(key));
  if (!area) return;
  area.classList.toggle('open');
  if (area.classList.contains('open')) {{
    area.querySelector('textarea').focus();
  }}
}}

function css_safe(s) {{ return s.replace(/[^a-zA-Z0-9]/g, '_'); }}

function updateAnnotation(persona, turn, text) {{
  const key = persona + ':' + turn;
  const btn = document.querySelector('[data-ann-key="' + key + '"]');
  if (text.trim()) {{
    annotations.set(key, text.trim());
    if (btn) btn.classList.add('has-note');
  }} else {{
    annotations.delete(key);
    if (btn) btn.classList.remove('has-note');
  }}
  document.getElementById('note-count').textContent = annotations.size;
  document.getElementById('export-btn').disabled = annotations.size === 0;
}}

async function exportAnnotations() {{
  if (annotations.size === 0) return;

  // Build notes summary
  const notes = [];
  annotations.forEach(function(note, key) {{
    const parts = key.split(':');
    const persona = parts[0];
    const turn = parseInt(parts[1]);
    notes.push({{persona: persona, turn: turn, note: note}});
  }});

  // Build full conversations for annotated personas only
  const annotatedPersonas = new Set(notes.map(function(n) {{ return n.persona; }}));
  const conversations = {{}};
  annotatedPersonas.forEach(function(persona) {{
    const turns = document.querySelectorAll('.turn[data-persona="' + persona + '"]');
    const conv = [];
    turns.forEach(function(el) {{
      const entry = {{
        turn: parseInt(el.dataset.turn),
        action: el.dataset.action || '',
        input: el.dataset.input || '',
        response: el.dataset.response || ''
      }};
      const key = persona + ':' + el.dataset.turn;
      if (annotations.has(key)) {{
        entry.note = annotations.get(key);
      }}
      conv.push(entry);
    }});
    conversations[persona] = conv;
  }});

  const data = {{
    report: REPORT_NAME,
    model: MODEL_KEY,
    character: CHARACTER,
    notes: notes,
    conversations: conversations
  }};

  const json = JSON.stringify(data, null, 2);
  const filename = REPORT_NAME + '.annotations.json';

  // If served over HTTP (via open_report), POST to server for direct save
  if (location.protocol === 'http:' || location.protocol === 'https:') {{
    try {{
      const resp = await fetch('/save-annotations', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: json
      }});
      const result = await resp.json();
      if (result.ok) {{
        lastSavedPath = result.path;
        navigator.clipboard.writeText(result.path).catch(function() {{}});
        flashSaved('Saved — path copied to clipboard');
        if (confirm('Annotations saved (path copied to clipboard):\\n' + result.path + '\\n\\nClose the review server?')) {{
          shutdownServer();
        }}
        return;
      }}
    }} catch (e) {{
      // Fall through to download
    }}
  }}

  // Fallback: browser download (file:// origin)
  const blob = new Blob([json], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  flashSaved('Downloaded — save to logs/annotated/');
}}

function shutdownServer() {{
  fetch('/shutdown', {{method: 'POST'}}).then(function() {{
    document.title = document.title + ' (server stopped)';
    const msg = lastSavedPath
      ? 'Server stopped — export saved to ' + lastSavedPath
      : 'Server stopped — you can close this tab';
    flashSaved(msg, true);
  }}).catch(function() {{}});
}}

// Show Done button when served over HTTP
if (location.protocol === 'http:' || location.protocol === 'https:') {{
  document.getElementById('done-btn').style.display = '';
}}

let _flashTimer = null;
function flashSaved(msg, persist) {{
  if (_flashTimer) {{ clearTimeout(_flashTimer); _flashTimer = null; }}
  const flash = document.getElementById('save-flash');
  flash.textContent = msg;
  flash.classList.add('show');
  if (!persist) _flashTimer = setTimeout(function() {{ flash.classList.remove('show'); _flashTimer = null; }}, 3000);
}}
</script>
"""


def json_esc(s: str) -> str:
    """JSON-encode a string for safe embedding in JS."""
    return json.dumps(s)


def _action_badge(action: str) -> str:
    """Return an HTML badge for the action type."""
    colors = {
        "message": "#58a6ff",
        "world": "#d29922",
        "beat": "#bc8cff",
        "see": "#7ee787",
        "nothing": "#8b949e",
    }
    color = colors.get(action, "#8b949e")
    return f'<span class="badge" style="background:{color}">{_esc(action)}</span>'


def generate_html_report(
    results: list[PersonaResult],
    model_key: str,
    model_name: str,
    character: str,
    timestamp: str,
) -> str:
    """Generate self-contained HTML report."""
    ts_display = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    # Summary stats
    total_turns = sum(len(r.turns) for r in results)
    total_stale = sum(1 for r in results for t in r.turns if t.stale_discard)
    errors = [r for r in results if r.error]

    personas_html = ""
    for pr in results:
        turns_html = ""
        for t in pr.turns:
            stale_tag = ' <span class="stale">STALE DISCARD</span>' if t.stale_discard else ""

            # Eval details
            eval_html = ""
            if t.eval_entry:
                e = t.eval_entry
                eval_html = (
                    f'<details class="eval-details"><summary>eval: {_esc(e.get("script_status", "?"))}</summary>'
                    f'<div class="eval-body">'
                    f'<div><b>thought:</b> {_esc(e.get("thought", ""))}</div>'
                    f'<div><b>gaze:</b> {_esc(e.get("gaze", ""))}</div>'
                    f'<div><b>expression:</b> {_esc(e.get("expression", ""))}</div>'
                    f'<div><b>status:</b> {_esc(e.get("script_status", ""))}</div>'
                )
                if e.get("plan_request"):
                    eval_html += f'<div><b>plan_request:</b> {_esc(e["plan_request"])}</div>'
                eval_html += "</div></details>"

            # Input display
            if t.action == "nothing":
                input_html = '<div class="idle">(does nothing)</div>'
            elif t.action == "beat":
                input_html = '<div class="beat-input">/beat</div>'
            elif t.action == "world":
                input_html = f'<div class="world-input">/world {_esc(t.input_text)}</div>'
            else:
                input_html = f'<div class="user-input">{_esc(t.input_text)}</div>'

            # Response
            resp_html = ""
            if t.response and t.response != "(idle)":
                resp_html = f'<div class="npc-response">{_esc(t.response)}</div>'

            persona_name = pr.persona.name
            ann_key = f"{persona_name}:{t.turn}"
            ann_key_safe = ann_key.replace(" ", "_").replace(":", "_")
            turns_html += (
                f'<div class="turn" data-persona="{_esc(persona_name)}" data-turn="{t.turn}"'
                f' data-action="{_esc(t.action)}" data-input="{_esc(t.input_text)}"'
                f' data-response="{_esc(t.response)}">'
                f'<div class="turn-header">Turn {t.turn + 1} {_action_badge(t.action)}{stale_tag}</div>'
                f'{input_html}{resp_html}{eval_html}'
                f'<button class="annotate-btn" data-ann-key="{_esc(ann_key)}"'
                f' onclick="toggleAnnotation(\'{_esc(persona_name)}\',{t.turn})">&#9998; annotate</button>'
                f'<div class="annotation-area" id="ann-{ann_key_safe}">'
                f'<textarea placeholder="Add a note..."'
                f' oninput="updateAnnotation(\'{_esc(persona_name)}\',{t.turn},this.value)"></textarea>'
                f'</div>'
                f'</div>'
            )

        error_html = ""
        if pr.error:
            error_html = f'<div class="error">Error: {_esc(pr.error)}</div>'

        # Boot plan info
        boot_html = ""
        for entry in pr.boot_log:
            if entry.get("type") == "plan":
                boot_html = (
                    f'<div class="boot-info">Boot plan: {entry["beats"]} beats '
                    f'({", ".join(_esc(i) for i in entry.get("intents", []))})</div>'
                )
                break

        personas_html += (
            f'<div class="persona-card">'
            f'<div class="persona-header">'
            f'<h2>{_esc(pr.persona.name)}</h2>'
            f'<span class="persona-desc">{_esc(pr.persona.description)}</span>'
            f'<span class="turn-count">{len(pr.turns)} turns</span>'
            f'</div>'
            f'{boot_html}{error_html}'
            f'<div class="conversation">{turns_html}</div>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Persona QA — {_esc(model_key)} — {timestamp}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, system-ui, 'Segoe UI', sans-serif;
    max-width: 1100px; margin: 2em auto; padding: 0 1em;
    background: #0d1117; color: #c9d1d9;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 0.3em; }}
  h2 {{ color: #c9d1d9; margin: 0; font-size: 1.2em; }}
  .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 1.5em; }}
  .summary {{
    display: flex; gap: 2em; padding: 1em; margin-bottom: 1.5em;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  }}
  .summary-stat {{ text-align: center; }}
  .summary-stat .num {{ font-size: 1.8em; font-weight: bold; color: #58a6ff; }}
  .summary-stat .label {{ font-size: 0.85em; color: #8b949e; }}
  .persona-card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    margin-bottom: 1.5em; overflow: hidden;
  }}
  .persona-header {{
    padding: 1em; border-bottom: 1px solid #30363d;
    display: flex; align-items: center; gap: 1em;
  }}
  .persona-desc {{ color: #8b949e; font-size: 0.9em; }}
  .turn-count {{
    margin-left: auto; background: #30363d; padding: 2px 10px;
    border-radius: 12px; font-size: 0.85em; color: #8b949e;
  }}
  .conversation {{ padding: 0.5em 1em; }}
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
  .idle {{ color: #8b949e; margin: 0.3em 0; padding-left: 1em; font-style: italic; }}
  .npc-response {{
    color: #7ee787; margin: 0.3em 0; padding-left: 1em;
    border-left: 3px solid #238636; padding: 0.3em 0.8em;
    background: #0d1117; border-radius: 0 4px 4px 0;
  }}
  .eval-details {{ margin: 0.3em 0 0 1em; }}
  .eval-details summary {{
    cursor: pointer; color: #8b949e; font-size: 0.85em;
  }}
  .eval-body {{
    padding: 0.5em; background: #0d1117; border-radius: 4px;
    font-size: 0.85em; margin-top: 0.3em;
  }}
  .eval-body div {{ margin: 0.2em 0; }}
  .boot-info {{
    padding: 0.5em 1em; color: #bc8cff; font-size: 0.85em;
    border-bottom: 1px solid #21262d;
  }}
  .error {{ padding: 0.5em 1em; color: #f85149; font-weight: bold; }}
</style></head><body>
<h1>Persona QA Report</h1>
<p class="meta">{ts_display} &middot; model: <b>{_esc(model_name)}</b> (<code>{_esc(model_key)}</code>) &middot; character: <b>{_esc(character)}</b></p>

<div class="summary">
  <div class="summary-stat"><div class="num">{len(results)}</div><div class="label">personas</div></div>
  <div class="summary-stat"><div class="num">{total_turns}</div><div class="label">total turns</div></div>
  <div class="summary-stat"><div class="num">{total_stale}</div><div class="label">stale discards</div></div>
  <div class="summary-stat"><div class="num">{len(errors)}</div><div class="label">errors</div></div>
</div>

{personas_html}
{annotation_assets(f"qa_personas_{model_key}_{timestamp}", model_key, character)}
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Persona-based subjective QA testing")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
        help=f"Chat model to test with (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--character", default=CHARACTER, help=f"Character to test (default: {CHARACTER})")
    parser.add_argument("--turns", type=int, default=None, help="Override turn count for all personas")
    parser.add_argument("--open", action="store_true", help="Open the HTML report in browser after generation")
    args = parser.parse_args()

    model_config = MODELS.get(args.model)
    if not model_config:
        print(f"Unknown model: {args.model}")
        return

    # Verify API key
    env = model_config.get("api_key_env", "")
    if env and not os.environ.get(env):
        print(f"Missing API key: {env}")
        return

    # Persona LLM: use big model if available, else chat model
    big_cfg = _get_big_model_config()
    persona_model_config = big_cfg or model_config

    print(f"Persona QA — model: {model_config['name']} ({args.model}), character: {args.character}")
    print(f"Persona LLM: {persona_model_config['name']}")
    print(f"Big model: {big_cfg['name'] if big_cfg else 'unavailable'}")
    print(f"Running {len(PERSONAS)} personas in parallel...\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.monotonic()

    results: list[PersonaResult] = [None] * len(PERSONAS)

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {}
        for i, persona in enumerate(PERSONAS):
            future = executor.submit(
                run_persona,
                persona,
                args.character,
                model_config,
                persona_model_config,
                args.turns,
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            persona = PERSONAS[idx]
            try:
                pr = future.result()
                results[idx] = pr
                stale = sum(1 for t in pr.turns if t.stale_discard)
                status = f"error: {pr.error}" if pr.error else f"{len(pr.turns)} turns, {stale} stale"
                print(f"  [{persona.name}] done — {status}")
            except Exception as e:
                results[idx] = PersonaResult(persona=persona, error=str(e))
                print(f"  [{persona.name}] FAILED — {e}")

    elapsed = time.monotonic() - start_time
    print(f"\nAll personas complete in {elapsed:.1f}s")

    # Generate HTML report
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "annotated").mkdir(exist_ok=True)
    html = generate_html_report(
        results=results,
        model_key=args.model,
        model_name=model_config["name"],
        character=args.character,
        timestamp=timestamp,
    )
    report_path = LOGS_DIR / f"qa_personas_{args.model}_{timestamp}.html"
    report_path.write_text(html)
    print(f"Report: {report_path}")

    if args.open:
        from character_eng.open_report import serve_report
        serve_report(report_path)


if __name__ == "__main__":
    main()
