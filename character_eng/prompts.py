import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"


def list_characters() -> list[str]:
    """Return sorted list of character directory names that contain a prompt.txt."""
    return sorted(
        d.name
        for d in CHARACTERS_DIR.iterdir()
        if d.is_dir() and (d / "prompt.txt").exists()
    )


def _read(path: Path) -> str:
    """Read a file, returning empty string if it doesn't exist."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def load_prompt(character: str) -> str:
    """Load and resolve a character's prompt template.

    Substitutes {{key}} macros in prompt.txt with:
      - global_rules -> prompts/global_rules.txt
      - character    -> prompts/characters/<name>/character.txt
      - scenario     -> prompts/characters/<name>/scenario.txt

    Unknown macros are left intact for debugging via /trace.
    """
    char_dir = CHARACTERS_DIR / character
    template = (char_dir / "prompt.txt").read_text()

    macros = {
        "global_rules": _read(PROMPTS_DIR / "global_rules.txt"),
        "character": _read(char_dir / "character.txt"),
        "scenario": _read(char_dir / "scenario.txt"),
    }

    def replace(m: re.Match) -> str:
        key = m.group(1)
        return macros[key] if key in macros else m.group(0)

    return re.sub(r"\{\{(\w+)\}\}", replace, template)
