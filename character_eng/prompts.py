import re
from pathlib import Path

from character_eng.creative import (
    character_asset_path,
    load_character_manifest,
    load_character_setup,
    load_prompt_registry,
    prompt_asset_path,
    resolve_character_scenario_path,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"


def list_characters() -> list[str]:
    """Return sorted list of character directory names that contain a prompt.txt."""
    names: list[str] = []
    for d in CHARACTERS_DIR.iterdir():
        if not d.is_dir():
            continue
        manifest = load_character_manifest(d.name, characters_dir=CHARACTERS_DIR)
        if not manifest.enabled:
            continue
        prompt_path = character_asset_path(d.name, "prompt", characters_dir=CHARACTERS_DIR)
        if prompt_path.exists():
            names.append(d.name)
    return sorted(names)


def _read(path: Path) -> str:
    """Read a file, returning empty string if it doesn't exist."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def prompt_source_paths(
    character: str,
    *,
    prompts_dir: Path = PROMPTS_DIR,
    characters_dir: Path = CHARACTERS_DIR,
    scenario_file: str | None = None,
) -> list[Path]:
    paths = [
        prompt_asset_path("global_rules", prompts_dir=prompts_dir, default_filename="global_rules.txt"),
        character_asset_path(character, "prompt", characters_dir=characters_dir),
        character_asset_path(character, "character", characters_dir=characters_dir),
    ]
    scenario_path = resolve_character_scenario_path(character, scenario_file, characters_dir=characters_dir)
    if scenario_path is not None:
        paths.append(scenario_path)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def prompt_source_signature(
    character: str,
    *,
    prompts_dir: Path = PROMPTS_DIR,
    characters_dir: Path = CHARACTERS_DIR,
    scenario_file: str | None = None,
) -> dict[str, int]:
    signature: dict[str, int] = {}
    for path in prompt_source_paths(
        character,
        prompts_dir=prompts_dir,
        characters_dir=characters_dir,
        scenario_file=scenario_file,
    ):
        try:
            signature[str(path)] = path.stat().st_mtime_ns
        except FileNotFoundError:
            signature[str(path)] = 0
    return signature


def mutable_prompt_inventory(
    character: str | None = None,
    *,
    prompts_dir: Path = PROMPTS_DIR,
    characters_dir: Path = CHARACTERS_DIR,
    scenario_file: str | None = None,
) -> list[dict]:
    registry = load_prompt_registry(prompts_dir=prompts_dir)
    items: list[dict] = []

    if character:
        for label, path, apply_mode, note in (
            (
                "Character prompt template",
                character_asset_path(character, "prompt", characters_dir=characters_dir),
                "refresh_or_continue",
                "Rebuilds the live base character prompt.",
            ),
            (
                "Character facts / voice",
                character_asset_path(character, "character", characters_dir=characters_dir),
                "refresh_or_continue",
                "Rebuilds the live base character prompt.",
            ),
            (
                "Scenario script",
                resolve_character_scenario_path(character, scenario_file, characters_dir=characters_dir),
                "restart_required",
                "State machine / stage graph changes still require /reload.",
            ),
        ):
            items.append({
                "label": label,
                "path": str(path),
                "kind": "character",
                "apply_mode": apply_mode,
                "apply_command": _apply_command_for_mode(apply_mode),
                "note": note,
                "exists": path.exists(),
            })

    for key in sorted(registry):
        path = prompt_asset_path(key, prompts_dir=prompts_dir, default_filename=f"{key}.txt")
        apply_mode = "next_call_auto"
        note = "Read from disk on the next model call."
        kind = "system"
        if key == "global_rules":
            apply_mode = "refresh_or_continue"
            note = "Part of the live base character prompt."
            kind = "base"
        elif key in {"vision_constant_questions", "vision_constant_sam_targets"}:
            apply_mode = "next_vision_cycle"
            note = "Read from disk on the next vision focus pass."
            kind = "vision"
        elif key in {"visual_focus", "vision_synthesis"}:
            apply_mode = "next_vision_cycle"
            note = "Read from disk on the next vision focus/synthesis pass."
            kind = "vision"
        items.append({
            "label": key.replace("_", " "),
            "path": str(path),
            "kind": kind,
            "apply_mode": apply_mode,
            "apply_command": _apply_command_for_mode(apply_mode),
            "note": note,
            "exists": path.exists(),
        })

    return items


def _apply_command_for_mode(mode: str) -> str:
    if mode == "next_vision_cycle":
        return "/refresh vision"
    if mode == "restart_required":
        return "/reload"
    return "/refresh"


def load_prompt(character: str, world_state=None, people_state=None, vision_context=None, scenario_file: str | None = None) -> str:
    """Load and resolve a character's prompt template.

    Substitutes {{key}} macros in prompt.txt with:
      - global_rules -> prompts/global_rules.txt
      - character    -> prompts/characters/<name>/character.txt
      - scenario     -> prompts/characters/<name>/scenario.txt
      - world        -> world_state.render() if provided
      - people       -> people_state.render() if provided
      - vision       -> vision_context.render_for_prompt() if provided

    Unknown macros are left intact for debugging via /trace.
    """
    template = character_asset_path(character, "prompt", characters_dir=CHARACTERS_DIR).read_text()
    setup = load_character_setup(character, characters_dir=CHARACTERS_DIR, scenario_file=scenario_file)

    macros = {
        "global_rules": _read(prompt_asset_path("global_rules", prompts_dir=PROMPTS_DIR, default_filename="global_rules.txt")),
        "character": _read(character_asset_path(character, "character", characters_dir=CHARACTERS_DIR)),
        "scenario": setup.premise if setup and setup.premise else _read(character_asset_path(character, "scenario", characters_dir=CHARACTERS_DIR)),
        "world": world_state.render() if world_state else "",
        "people": people_state.render() if people_state else "",
        "vision": vision_context.render_for_prompt() if vision_context else "",
    }

    def replace(m: re.Match) -> str:
        key = m.group(1)
        return macros[key] if key in macros else m.group(0)

    return re.sub(r"\{\{(\w+)\}\}", replace, template)
