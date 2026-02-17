import re
import json
from datetime import datetime
from pathlib import Path
from collections import deque

import gymnasium as gym
import ollama
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage

# --- Configuration ---
MODEL_NAME = 'gpt-oss:120b'
HISTORY_LENGTH = 10
NOTES_LENGTH = 5  # How many notes the LLM can keep at once
MAX_CHAT_HISTORY = 20  # Keeps the last 10 conversational turns (User + Assistant) to prevent context overflow
LOG_DIR = Path("logs")
LLM_DISPLAY_LINES = 25

# --- State Variables ---
action_history = deque(maxlen=HISTORY_LENGTH)
notes_history = deque(maxlen=NOTES_LENGTH)
current_obs = ""
system_feedback = ""  # Used for out-of-band feedback (like confirming a note)
llm_response_text = ""
run_log = []


def build_layout() -> Layout:
    """Constructs the TUI Dashboard."""
    layout = Layout()

    layout.split_column(
        Layout(name="state", ratio=1),
        Layout(name="llm", ratio=2)
    )

    # Split the Left Column into Obs and Tracking
    layout["state"].split_column(
        Layout(name="obs", ratio=2),
        Layout(name="tracking", ratio=1)
    )

    # Split the Tracking section into History and Notes
    layout["tracking"].split_row(
        Layout(name="history", ratio=1),
        Layout(name="notes", ratio=2)
    )

    # Observation Panel
    layout["obs"].update(
        Panel(current_obs, title="[bold cyan]Current Observation", border_style="cyan")
    )

    # History Panel
    history_text = "\n".join(action_history) if action_history else "No actions yet."
    layout["history"].update(
        Panel(history_text, title="[bold magenta]Recent Actions", border_style="magenta")
    )

    # Notes Panel
    notes_text = "\n".join([f"- {n}" for n in notes_history]) if notes_history else "No notes written."
    layout["notes"].update(
        Panel(notes_text, title="[bold yellow]Agent Scratchpad", border_style="yellow")
    )

    # LLM Panel
    lines = llm_response_text.split('\n')
    if len(lines) > LLM_DISPLAY_LINES:
        display_text = "\n".join(lines[-LLM_DISPLAY_LINES:])
    else:
        display_text = llm_response_text

    display_text = display_text.replace("[", "\\[")

    layout["llm"].update(
        Panel(display_text, title=f"[bold green]LLM Internal Monologue ({MODEL_NAME})", border_style="green")
    )

    return layout


def parse_commands(text: str) -> list[tuple[str, str]]:
    """Parses the LLM's response for notes, deletes, and actions using dedicated XML tags."""
    commands = []

    # 1. Extract Notes
    note_matches = re.finditer(r'<note>\s*(.*?)\s*</note>', text, re.IGNORECASE | re.DOTALL)
    for match in note_matches:
        commands.append(("NOTE", match.group(1).strip()))

    # 2. Extract Deletes
    delete_matches = re.finditer(r'<delete>\s*(\d+)\s*</delete>', text, re.IGNORECASE)
    for match in delete_matches:
        commands.append(("DELETE", match.group(1).strip()))

    # 3. Extract Actions (Movements and Utility)
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if action_match:
        action_str = action_match.group(1).strip()
        parts = re.split(r'[.,]', action_str)
        for part in parts:
            clean = part.strip().upper()
            if clean in ["NORTH", "SOUTH", "EAST", "WEST", "LOOK", "RESET", "QUIT", "HINT"]:
                commands.append(("ACTION", clean))
    else:
        # Fallback to standard movement extraction if no action tag is found
        matches = re.findall(r'\b(NORTH|SOUTH|EAST|WEST|LOOK|RESET|QUIT|HINT)\b', text, re.IGNORECASE)
        if matches:
            commands.append(("ACTION", matches[-1].upper()))

    # If it failed to do *anything* useful, flag it
    if not commands:
        commands.append(("ACTION", "INVALID"))

    return commands


def save_log_entry(log_file, entry):
    """Append a single log entry to the JSONL file immediately."""
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def main():
    global current_obs, system_feedback, llm_response_text, action_history, notes_history, run_log

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.jsonl"

    run_start_entry = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "type": "run_start"
    }
    run_log.append(run_start_entry)
    save_log_entry(log_file, run_start_entry)

    env = BoltCrypt(generator_config={
        'min_dist': 3,
        'mean_rooms': 6,
        "std_rooms": 0,
        'puzzle_density': 0,
        'key_puzzle_prob': 0,
        'puzzle_required': False,
        'allowed_puzzles': ['boulder'],
        'min_room_dim': 5,
        'max_room_dim': 7,
    }, render_mode='human')
    env = NaturalLanguage(env)

    obs, info = env.reset()
    current_obs = obs
    done = False

    system_prompt_template = (
        "You are an AI adventurer playing a text-based dungeon crawler. "
        "Find the room with the exit tile and step on it to win. "
        "Reason about your surroundings and the puzzles within and plan your route. "
        "Take notes to remember where you've been.  Rooms look the same but note differences in door positions.  "
        "Ask for a HINT if you feel stuck. If you wan to give up, type QUIT.  "
        "W = walls, D = doors, K = keys, B = boulders, E = exit, P = Pressure Plate, S = Sailing Stone, Y = You.  "
        "Two items on a single tile will attempt to be displayed [B+P] or [Y+P].  "
        "AVAILABLE COMMANDS:\n"
        "1. Utility: <action>LOOK</action> (re-describe room), <action>RESET</action> (new dungeon), <action>HINT</action> (get puzzle hint)\n"
        "2. Actions: <action>NORTH, EAST, SOUTH, WEST</action> (execute in sequence, separated by commas)\n"
        "3. Take Note: <note>Found locked door to the West, need red key.</note> This saves to your scratchpad.\n"
        "4. Delete Note: <delete>0</delete> to delete the note at index 0.\n\n"
        "IMPORTANT: You can take a note and perform an action in the same turn. Always wrap your final choices in their respective tags."
    )

    with Live(build_layout(), refresh_per_second=12, screen=True) as live:
        step_count = 0
        deque_action_history = deque(maxlen=5)

        # Initialize our persistent chat history
        chat_history = deque(maxlen=MAX_CHAT_HISTORY)

        while not done:
            step_count += 1
            llm_response_text = ""
            live.update(build_layout())

            history_str = ", ".join(deque_action_history) if deque_action_history else "None"
            notes_str = "\n".join(
                [f"{i} - {n}" for i, n in enumerate(notes_history)]) if notes_history else "No notes active."

            # Construct the current system message
            current_system_message = {
                'role': 'system',
                'content': f"{system_prompt_template}\n\nRecent Actions: {history_str}\n\nActive Notes:\n{notes_str}"
            }

            # Append the system feedback (like "Note saved") to the current observation temporarily
            turn_prompt = f"{current_obs}\n{system_feedback}".strip()
            current_user_message = {'role': 'user', 'content': turn_prompt}

            # Build the payload: System prompt + rolling chat history + current turn
            messages = [current_system_message] + list(chat_history) + [current_user_message]

            response_stream = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                options={'temperature': 0.1}
            )

            in_thinking = False
            raw_assistant_content = ""

            for chunk in response_stream:
                if getattr(chunk.message, 'thinking', None):
                    if not in_thinking:
                        in_thinking = True
                        llm_response_text += '--- SYNAPSES FIRING ---\n'
                    llm_response_text += chunk.message.thinking

                elif chunk.message.content:
                    if in_thinking:
                        llm_response_text += '\n--- CONCLUSION ---\n'
                        in_thinking = False

                    # Accumulate raw text for the history buffer
                    raw_assistant_content += chunk.message.content
                    llm_response_text += chunk.message.content

                live.update(build_layout())

            # Update conversational history memory
            chat_history.append(current_user_message)
            chat_history.append({'role': 'assistant', 'content': raw_assistant_content})

            # Clear out-of-band feedback for the next turn
            system_feedback = ""

            commands = parse_commands(raw_assistant_content)

            for cmd_type, cmd_value in commands:

                if cmd_type == "NOTE":
                    notes_history.append(cmd_value)
                    action_history.appendleft(f"Step {step_count}: [bold cyan]NOTE TAKEN[/]")
                    deque_action_history.append("Wrote Note")
                    system_feedback += "\n> System: Note successfully added to your scratchpad."

                    note_entry = {
                        "type": "note",
                        "step": step_count,
                        "note_content": cmd_value,
                        "llm_response": raw_assistant_content
                    }
                    run_log.append(note_entry)
                    save_log_entry(log_file, note_entry)

                elif cmd_type == "DELETE":
                    try:
                        notes_history.remove(notes_history[int(cmd_value)])
                        action_history.appendleft(f"Step {step_count}: [bold cyan]NOTE DELETED[/]")
                        deque_action_history.append("Deleted Note")
                        system_feedback += "\n> System: You tear a page from your notepad."
                    except (IndexError, ValueError):
                        system_feedback += f"\n> System: Failed to delete note at index {cmd_value}."

                    delete_entry = {
                        "type": "delete",
                        "step": step_count,
                        "note_number": cmd_value,
                        "llm_response": raw_assistant_content
                    }
                    run_log.append(delete_entry)
                    save_log_entry(log_file, delete_entry)

                elif cmd_type == "ACTION":
                    action = cmd_value
                    deque_action_history.append(action)

                    if action in ["NORTH", "SOUTH", "EAST", "WEST", "LOOK", "RESET", "QUIT", "HINT"]:
                        obs, reward, terminated, truncated, info = env.step(action)
                        env.render()
                        done = terminated or truncated
                        current_obs = obs
                        action_history.appendleft(f"Step {step_count}: [bold yellow]{action}[/] (R: {reward})")

                        step_entry = {
                            "type": "step",
                            "step": step_count,
                            "observation": obs,
                            "action": action,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "llm_response": raw_assistant_content
                        }
                        run_log.append(step_entry)
                        save_log_entry(log_file, step_entry)

                        if done:
                            break
                    else:
                        action_history.appendleft(f"Step {step_count}: [bold red]FAIL ({action})[/]")
                        system_feedback += f"\n> System: Invalid action '{action}'. Use NORTH, SOUTH, EAST, WEST, LOOK, RESET, or HINT."

                        error_entry = {
                            "type": "error",
                            "step": step_count,
                            "observation": current_obs,
                            "action": action,
                            "error": "Invalid action format",
                            "llm_response": raw_assistant_content
                        }
                        run_log.append(error_entry)
                        save_log_entry(log_file, error_entry)

            live.update(build_layout())

        # --- Post-Game Rating Loop ---
        llm_response_text = "Game Over. Requesting performance rating...\n\n"
        live.update(build_layout())

        # Detect if the agent quit early
        quit_early = any(entry.get("action") == "QUIT" for entry in run_log if entry.get("type") == "step")

        # Include chat history here so the LLM actually remembers how long it was in the dungeon!
        rating_prompt = "How was your time in the dungeon?"
        if quit_early:
            rating_prompt = "You chose to quit the game early. Why did you decide to quit? How was your time in the dungeon?"

        rating_messages = [
                              {'role': 'system', 'content': f"The game has ended. Final observation:\n{current_obs}"}
                          ] + list(chat_history) + [
                              {'role': 'user', 'content': rating_prompt}
                          ]

        rating_stream = ollama.chat(
            model=MODEL_NAME,
            messages=rating_messages,
            stream=True,
            options={'temperature': 0.3}
        )

        in_thinking = False
        final_rating_content = ""
        for chunk in rating_stream:
            if getattr(chunk.message, 'thinking', None):
                if not in_thinking:
                    in_thinking = True
                    llm_response_text += '--- SYNAPSES FIRING ---\n'
                llm_response_text += chunk.message.thinking
            elif chunk.message.content:
                if in_thinking:
                    llm_response_text += '\n--- RATING ---\n'
                    in_thinking = False
                llm_response_text += chunk.message.content
                final_rating_content += chunk.message.content

            live.update(build_layout())

        notes_entry = {
            "type": "notes",
            "response": list(notes_history)
        }
        run_log.append(notes_entry)
        save_log_entry(log_file, notes_entry)

        rating_entry = {
            "type": "rating",
            "response": final_rating_content,
            "quit_early": quit_early
        }
        run_log.append(rating_entry)
        save_log_entry(log_file, rating_entry)

    print(f"\nSimulation complete. Log saved to: {log_file}")


if __name__ == "__main__":
    main()