import re
from pathlib import Path

from character_eng.creative import character_asset_path, load_character_manifest, load_character_setup, prompt_asset_path

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
