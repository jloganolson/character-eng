from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterManifest:
    enabled: bool = True
    prompt: str = "prompt.txt"
    character: str = "character.txt"
    scenario: str = "scenario.txt"
    world_static: str = "world_static.txt"
    world_dynamic: str = "world_dynamic.txt"
    default_scenario_script: str = "scenario_script.toml"


@dataclass(frozen=True)
class SituationSetup:
    premise: str = ""
    static_facts: tuple[str, ...] = ()
    dynamic_facts: tuple[str, ...] = ()
    gaze_targets: tuple[str, ...] = ()


def load_character_manifest(character: str, *, characters_dir: Path) -> CharacterManifest:
    char_dir = characters_dir / character
    manifest_path = char_dir / "character_manifest.toml"
    if not manifest_path.exists():
        return CharacterManifest()
    data = tomllib.loads(manifest_path.read_text())
    files = data.get("files", {})
    return CharacterManifest(
        enabled=bool(data.get("meta", {}).get("enabled", True)),
        prompt=str(files.get("prompt", "prompt.txt")),
        character=str(files.get("character", "character.txt")),
        scenario=str(files.get("scenario", "scenario.txt")),
        world_static=str(files.get("world_static", "world_static.txt")),
        world_dynamic=str(files.get("world_dynamic", "world_dynamic.txt")),
        default_scenario_script=str(files.get("default_scenario_script", "scenario_script.toml")),
    )


def character_asset_path(
    character: str,
    asset: str,
    *,
    characters_dir: Path,
) -> Path:
    manifest = load_character_manifest(character, characters_dir=characters_dir)
    filename = getattr(manifest, asset)
    return characters_dir / character / filename


def load_prompt_registry(*, prompts_dir: Path) -> dict[str, str]:
    registry_path = prompts_dir / "prompt_registry.toml"
    if not registry_path.exists():
        return {}
    data = tomllib.loads(registry_path.read_text())
    prompts = data.get("prompts", {})
    return {str(key): str(value) for key, value in prompts.items()}


def prompt_asset_path(
    key: str,
    *,
    prompts_dir: Path,
    default_filename: str | None = None,
) -> Path:
    registry = load_prompt_registry(prompts_dir=prompts_dir)
    filename = registry.get(key, default_filename or key)
    return prompts_dir / filename


def load_prompt_asset(
    key: str,
    *,
    prompts_dir: Path,
    default_filename: str | None = None,
) -> str:
    return prompt_asset_path(key, prompts_dir=prompts_dir, default_filename=default_filename).read_text()


def resolve_character_scenario_path(
    character: str,
    filename: str | None = None,
    *,
    characters_dir: Path,
) -> Path:
    manifest = load_character_manifest(character, characters_dir=characters_dir)
    requested = Path(filename or manifest.default_scenario_script)
    if requested.name in {"", ".", ".."} or requested.is_absolute() or ".." in requested.parts:
        raise ValueError(f"invalid scenario filename: {filename!r}")
    return (characters_dir / character / requested).resolve()


def load_character_setup(character: str, *, characters_dir: Path, scenario_file: str | None = None) -> SituationSetup | None:
    path = resolve_character_scenario_path(character, scenario_file, characters_dir=characters_dir)
    if not path.exists():
        return None
    data = tomllib.loads(path.read_text())
    setup = data.get("setup", {})
    return SituationSetup(
        premise=str(setup.get("premise", "")).strip(),
        static_facts=tuple(str(item).strip() for item in setup.get("static_facts", []) if str(item).strip()),
        dynamic_facts=tuple(str(item).strip() for item in setup.get("dynamic_facts", []) if str(item).strip()),
        gaze_targets=tuple(str(item).strip() for item in setup.get("gaze_targets", []) if str(item).strip()),
    )
