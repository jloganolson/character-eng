#!/bin/bash
# Create a new character skeleton.
# Usage: ./scripts/new_character.sh <name>
#
# Creates prompts/characters/<name>/ with all required files.
# Edit the files to customize your character.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <character_name>"
    echo "Example: $0 alice"
    exit 1
fi

NAME="$1"
DIR="prompts/characters/$NAME"

if [ -d "$DIR" ]; then
    echo "Error: $DIR already exists"
    exit 1
fi

mkdir -p "$DIR/sims"

# prompt.txt
cat > "$DIR/prompt.txt" << 'PROMPT'
{{global_rules}}

## Character
{{character}}

## Current Scenario
{{scenario}}

## World State
{{world}}

## People Present
{{people}}

## What You See
{{vision}}

Play this character in a conversation with the player. Respond to what they say and do as this character would, staying true to the personality and situation described above.
PROMPT

# character.txt
cat > "$DIR/character.txt" << CHARACTER
You are $NAME, a robot head mounted on a table. You have eyes and a speaker but no body, arms, or hands.

[Describe personality, quirks, how they speak]

[Give 2-3 example responses to show voice/tone]

Long-term goal: [What drives them fundamentally]
Short-term goal: [What they're trying to do right now]
CHARACTER

# scenario.txt
cat > "$DIR/scenario.txt" << SCENARIO
[Describe the current situation. Where is the character? What are they doing?
What's the immediate context? What objects or props are around them?]
SCENARIO

# world_static.txt
cat > "$DIR/world_static.txt" << STATIC
$NAME is a robot head mounted on a table
$NAME has eyes and a speaker but no body, arms, or hands
[Add 5-10 permanent facts about the scene, one per line]
STATIC

# world_dynamic.txt
cat > "$DIR/world_dynamic.txt" << DYNAMIC
It is afternoon
No one is nearby — $NAME is alone
[Add 3-5 mutable starting facts, one per line]
DYNAMIC

# scenario_script.toml
cat > "$DIR/scenario_script.toml" << 'TOML'
[scenario]
name = "CHARACTER_NAME Scene"
start = "idle"

[[stage]]
name = "idle"
goal = "CHARACTER_NAME is alone. [What do they do while waiting?]"

[[stage.exit]]
condition = "Someone approaches or shows interest"
goto = "engaged"
label = "notices"

[[stage]]
name = "engaged"
goal = "Someone is here. [What does CHARACTER_NAME try to do?]"

[[stage.exit]]
condition = "The interaction concludes naturally"
goto = "idle"
label = "done"
TOML

# Replace CHARACTER_NAME in scenario_script.toml
sed -i "s/CHARACTER_NAME/$NAME/g" "$DIR/scenario_script.toml"

# Sample sim script
cat > "$DIR/sims/hello.sim.txt" << SIM
# Hello sim: basic approach and greeting
# Format: time_offset | description
# Quoted descriptions become user dialogue

0.0 | A person walks up
3.0 | "Hey there"
6.0 | "What's this?"
9.0 | "Interesting"
12.0 | "Well, nice meeting you"
15.0 | The person walks away
SIM

echo "Created character skeleton at $DIR/"
echo ""
echo "Files to edit:"
echo "  $DIR/character.txt      — personality and goals"
echo "  $DIR/scenario.txt       — current situation"
echo "  $DIR/world_static.txt   — permanent scene facts"
echo "  $DIR/world_dynamic.txt  — mutable starting state"
echo "  $DIR/scenario_script.toml — stage graph"
echo "  $DIR/sims/hello.sim.txt — test sim script"
echo ""
echo "Test your character:"
echo "  uv run -m character_eng --character $NAME"
echo "  uv run -m character_eng --character $NAME --sim hello"
