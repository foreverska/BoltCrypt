import gymnasium as gym
import numpy as np
import random
import zlib


class NaturalLanguage(gym.Wrapper):
    """
    Wrapper that converts observations to natural language descriptions.
    map_size: Size of the dungeon map (default: 5x5), None if no map
    room_ident: If True, generates a deterministic 3-word thematic identifier for each room.

    Designed for LLM agents, this wrapper:
    - Replaces grid observations with rich text descriptions
    - Accepts text commands (NORTH, SOUTH, EAST, WEST, LOOK, RESET, EXIT)
    - Provides contextual flavor text and tactical information
    - Includes event logging for game state changes
    """

    def __init__(self, env, map_size: int | None = 5, room_ident: bool = True):
        """
        Initialize the natural language wrapper.

        Args:
            env: BoltCrypt environment instance to wrap
        """
        super().__init__(env)
        self.last_gx, self.last_gy = None, None
        self.last_solved = True
        self.last_obs = None
        self.just_entered_room = True  # trigger flavor text on first spawn
        self.room_ident = room_ident

        if map_size is not None and map_size % 2 == 0:
            raise ValueError("Map size must be odd.")
        self.map_size = map_size

        # Command mapping
        self.mapping = {
            'north': 0, 'n': 0, 'up': 0,
            'south': 1, 's': 1, 'down': 1,
            'east': 2, 'e': 2, 'right': 2,
            'west': 3, 'w': 3, 'left': 3
        }

        # Room Fingerprint Vocabulary
        self.word_pool = [
            "VOID", "GEAR", "BLOOD", "COBALT", "SILENCE", "ECHO", "ICHOR", "RUST",
            "CROWN", "SHADOW", "BONE", "SULFUR", "MERCURY", "SPIRE", "THORN", "VALLEY",
            "GHOST", "PULSE", "CRISMAL", "VESSEL", "RAZOR", "OBELISK", "EMBER", "FROST"
        ]

        # Flavor Text Library
        self.flavor_texts = {
            "BOULDER": [
                "The floor is scarred with deep grooves. Heavy boulders sit silently, waiting to be moved.",
                "You hear the grinding of stone on stone. Pressure plates line the floor.",
                "A puzzle of weight and leverage lies before you.",
                "A one tile walkway around the edge of the room surrounds a pit of boulders and pressure plates."
            ],
            "STONE": [
                "The air is filled with a low, unnerving hum. Stones slide across the floor of their own accord.",
                "Watch your step. The rocks here are alive and move with crushing force.",
                "An unnatural wind blows through this chamber, pushing debris in random patterns."
            ],
            "WARP_CYCLE": [
                "The geometry of this room is broken. The floor tiles ripple like water.",
                "Space folds in on itself here. One step forward might take you ten steps back.",
                "A nausea-inducing aura permeates the room. Teleportation magic is active."
            ],
            "WARP_RND": [
                "Chaos reigns here. The very fabric of reality is unstable.",
                "Do not trust your eyes. The ground dissolves and reforms constantly.",
                "A chaotic rift chamber. Every step is a gamble."
            ],
            "NONE": [
                "A quiet, dusty chamber. The air is still.",
                "The room is empty, save for the echoes of your own footsteps.",
                "Shadows stretch long across the cold flagstones."
            ]
        }

    def reset(self, **kwargs):
        """
        Reset the environment and return initial text description.

        Args:
            **kwargs: Arguments passed to base environment reset

        Returns:
            tuple: (text description, info dict)
        """
        obs, info = self.env.reset(**kwargs)
        self.last_gx, self.last_gy = obs['global_pos']
        self.last_solved = self.env.curr_room.check_solved()
        self.last_obs = obs
        self.just_entered_room = True

        intro_text = (
            "*** SYSTEM: PROTOCOL INITIATED ***\n"
            "MISSION: Escape the Labyrinth.\n"
            "INSTRUCTIONS: The room is a grid. The dungeon is a set of connected rooms. Solve puzzles to unlock doors. Doors not locked by a puzzle may need a key. Reach the Exit.\n"
            "Commands: NORTH, SOUTH, EAST, WEST, LOOK, RESET, HINT, QUIT.\n"
            "Commands may be chained (e.g., 'north, east, look') or used alone.\n"
        )
        return self._generate_text(obs, 0, False, intro_text), info

    def step(self, action_input: str):
        """
        Process a text command or numeric action and return text description.

        Accepted commands:
        - Directional: NORTH/N/UP, SOUTH/S/DOWN, EAST/E/RIGHT, WEST/W/LEFT
        - Utility: LOOK (re-describe room), RESET (new dungeon), EXIT/QUIT (terminate)
        - Zork-style: Multiple commands separated by commas or periods (e.g., "north, east, look")

        Args:
            action_input: String command, integer action (0-3), or comma/period-separated command list

        Returns:
            tuple: (text description, reward, done, truncated, info)
        """
        # 1. PARSE - Handle Zork-style command lists
        if isinstance(action_input, str):
            # Check if this is a command list (contains comma or period separators)
            if ',' in action_input or (action_input.count('.') > 0 and action_input.strip()[-1] != '.'):
                # Split on commas first, then on periods within each segment
                commands = []
                for segment in action_input.split(','):
                    # Split on periods but ignore trailing period (single command case)
                    parts = [p.strip() for p in segment.split('.') if p.strip()]
                    commands.extend(parts)

                # Execute commands sequentially
                all_outputs = []
                total_reward = 0
                for i, cmd_str in enumerate(commands):
                    obs_text, reward, done, trunc, info = self.step(cmd_str)
                    all_outputs.append(f"Command {i + 1}: {cmd_str.upper()}\n{obs_text}")
                    total_reward += reward
                    if done or trunc:
                        return "\n\n".join(all_outputs), total_reward, done, trunc, info
                return "\n\n".join(all_outputs), total_reward, False, False, {}

            # Single command processing
            cmd = action_input.lower().strip().rstrip('.')

            if cmd in ['quit']:
                return "System: Session Terminated.", 0, True, False, {}
            if cmd == 'reset':
                obs, _ = self.env.reset()
                self.last_gx, self.last_gy = obs['global_pos']
                self.last_solved = True
                self.last_obs = obs
                self.just_entered_room = True
                return self._generate_text(obs, 0, False, "System: World Reset."), 0, False, False, {}
            if cmd == 'look':
                self.just_entered_room = True  # Force full description
                return self._generate_text(self.last_obs, 0, False,
                                           "You take a moment to survey the room."), 0, False, False, {}
            if cmd == 'hint':
                hint_text = self._generate_hint()
                return hint_text, 0, False, False, {}

            if cmd in self.mapping:
                action = self.mapping[cmd]
            else:
                return f"System: Invalid command '{cmd}'.", 0, False, False, {}
        else:
            action = action_input

        # 2. PRE-COMPUTE
        prev_pos = self.last_obs['agent_pos']
        prev_boulders = [list(b) for b in self.env.curr_room.boulders]

        # 3. STEP
        obs, reward, done, trunc, info = self.env.step(action)
        self.last_obs = obs

        # 4. EVENT LOGGING
        events = []

        # Check Room Transition
        curr_gx, curr_gy = obs['global_pos']
        if curr_gx != self.last_gx or curr_gy != self.last_gy:
            self.just_entered_room = True  # Flag for flavor text
            dx, dy = curr_gx - self.last_gx, curr_gy - self.last_gy

            # Determine entry direction
            if dy == 1:
                entered_from = "SOUTH"  # Moved North
            elif dy == -1:
                entered_from = "NORTH"  # Moved South
            elif dx == 1:
                entered_from = "WEST"  # Moved East
            elif dx == -1:
                entered_from = "EAST"  # Moved West
            else:
                entered_from = "UNKNOWN"

            events.append(f"You step through the door and enter a new room from its {entered_from}.")
            self.last_gx, self.last_gy = curr_gx, curr_gy
            self.last_solved = self.env.curr_room.check_solved()
        else:
            self.just_entered_room = False
            # Movement checks
            new_pos = obs['agent_pos']
            if not np.array_equal(prev_pos, new_pos):
                dist = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                if dist > 1:
                    events.append("WARP! The world twists. You reappear elsewhere!")
                else:
                    events.append(f"You moved 1 step {action_input}.")
            elif reward < 0:
                events.append(f"BLOCKED. You bump into a solid obstacle trying to move {action_input}.")

            # Interaction checks
            curr_boulders = self.env.curr_room.boulders
            for i, b in enumerate(curr_boulders):
                if b != prev_boulders[i]:
                    events.append("You heave the massive BOULDER forward.")
                    if tuple(b) in self.env.curr_room.switches:
                        events.append("CLICK! It locks into the pressure plate.")

        # Key check
        if reward == 1.0 and obs['inventory'] == 1:
            events.append("SUCCESS! You acquired the GOLDEN KEY.")

        # Solved check
        is_solved = self.env.curr_room.check_solved()
        if is_solved and not self.last_solved:
            events.append("MECHANISM: Gears grind and locks disengage. The exits are open.")
        self.last_solved = is_solved

        event_str = " ".join(events) if events else "Time passes..."

        if done:
            return f"{event_str}\n\n*** VICTORY! YOU HAVE ESCAPED! ***", reward, done, trunc, info

        return self._generate_text(obs, reward, done, event_str), reward, done, trunc, info

    def _get_flavor_text(self, room):
        """
        Generate atmospheric flavor text based on room puzzle type.
        """
        p_type = room.puzzle_type.name
        if p_type not in self.flavor_texts: p_type = "NONE"
        return random.choice(self.flavor_texts[p_type])

    def _get_room_fingerprint(self, gx, gy):
        """
        Generate a deterministic 3-word phrase for a given room coordinate.
        """
        seed_val = zlib.adler32(f"{gx},{gy}".encode())
        rng = random.Random(seed_val)
        chosen = rng.sample(self.word_pool, 3)
        return " ".join(chosen)

    def _generate_text(self, obs, reward, done, event_text):
        """
        Generate complete text description of current game state.
        """
        room = self.env.curr_room
        lx, ly = obs['agent_pos']
        gx, gy = obs['global_pos']
        w, h = room.w, room.h

        out = []

        # --- 1. EVENT LOG (Immediate Feedback) ---
        out.append(f"\n> {event_text}")

        # --- 2. NARRATIVE (On Entry) ---
        if self.just_entered_room:
            out.append(f"\n--- NEW CHAMBER DISCOVERED ---")
            out.append(f"\"{self._get_flavor_text(room)}\"")

            # THE FINGERPRINT
            if self.room_ident:
                fingerprint = self._get_room_fingerprint(gx, gy)
                scrawl_variants = [
                    f"Scrawled in charcoal on the wall are the words: [ {fingerprint} ]",
                    f"You notice three words etched into the floor: [ {fingerprint} ]",
                    f"A series of strange runes on the ceiling translate to: [ {fingerprint} ]",
                    f"Faded blood forms three distinct words near the center: [ {fingerprint} ]"
                ]
                variant_idx = zlib.adler32(f"{gx},{gy},var".encode()) % len(scrawl_variants)
                out.append(scrawl_variants[variant_idx])

            if room.is_locked and not room.has_puzzle:
                out.append("NOTICE: The door ahead is locked. You need a key.")

        # --- 3. TACTICAL OVERVIEW (Textual) ---
        out.append(f"\n--- SITUATION REPORT ---")
        out.append(f"LOCATION: {w}x{h} tile room of a larger labyrinth, current tile in room ({lx}, {ly})")

        # DOORS
        door_descs = []

        for d, off in room.doors.items():
            if d.name == "NORTH":
                door_x, door_y = off, h - 1
            elif d.name == "SOUTH":
                door_x, door_y = off, 0
            elif d.name == "EAST":
                door_x, door_y = w - 1, off
            elif d.name == "WEST":
                door_x, door_y = 0, off
            else:
                continue

            dx = door_x - lx
            dy = door_y - ly
            direction_text = self._get_relative_dir(dx, dy)
            state = "(OPEN)" if (room.is_solved or room.puzzle_type.name == "NONE") else "(LOCKED)"
            door_descs.append(f"A door is {direction_text} from you: {state}")

        out.append("EXITS: " + ", ".join(door_descs))

        # INTERESTING OBJECTS (Relative to Player)
        objects = []

        # THE EXIT
        if room.is_exit:
            ey, ex = room.h // 2, room.w // 2
            dx, dy = ex - lx, ey - ly
            exit_dir = self._get_relative_dir(dx, dy)
            exit_desc = (
                f"THE ARCHWAY TO FREEDOM: A jagged, pulsating rift has torn through the center of the floor. "
                f"A massive, buzzing neon sign flickers overhead, casting a sickly pink glow and screaming 'EXIT' "
                f"in ancient, humming circuitry. It is located {exit_dir}."
            )
            objects.append(exit_desc)

        # Switches
        for (sy, sx) in room.switches:
            covered = any(b[0] == sy and b[1] == sx for b in room.boulders)
            status = "ACTIVATED" if covered else "EMPTY"
            dx, dy = sx - lx, sy - ly
            dir_str = self._get_relative_dir(dx, dy)
            objects.append(f"Pressure Plate ({status}): {dir_str}")

        # Columns
        if hasattr(room, 'columns'):
            for i, b in enumerate(room.columns):
                by, bx = b[0], b[1]
                dx, dy = bx - lx, by - ly
                dir_str = self._get_relative_dir(dx, dy)
                objects.append(f"Column: {dir_str}")

        # Boulders
        for i, b in enumerate(room.boulders):
            by, bx = b[0], b[1]
            dx, dy = bx - lx, by - ly
            dir_str = self._get_relative_dir(dx, dy)
            objects.append(f"Heavy Boulder: {dir_str}")

        # Key
        if room.has_key and obs['inventory'] == 0:
            ky, kx = room.key_pos
            dx, dy = kx - lx, ky - ly
            dir_str = self._get_relative_dir(dx, dy)
            objects.append(f"GOLDEN KEY: {dir_str}")

        if objects:
            out.append("NEARBY: " + "; ".join(objects))

        # --- 4. THE HUD (Grid) ---
        if self.map_size is not None:
            out.append(f"\n--- ROOM TILES AROUND ADVENTURER (each [] a room tile) ---")
            grid_viz = []
            radius = self.map_size // 2

            for dy in range(radius, -radius - 1, -1):
                row_str = ""
                for dx in range(-radius, radius + 1):
                    ny, nx = ly + dy, lx + dx

                    if dx == 0 and dy == 0:
                        on_switch = (ny, nx) in room.switches
                        if on_switch:
                            row_str += "[Y+P]"
                        else:
                            row_str += "[ Y ]"
                        continue

                    if not (0 <= ny < h and 0 <= nx < w):
                        row_str += "[UNK]"
                        continue

                    tile = room.grid_array[ny, nx]
                    is_boulder = any(b[0] == ny and b[1] == nx for b in room.boulders)
                    is_switch = (ny, nx) in room.switches
                    is_stone = any(s[0] == ny and s[1] == nx for s in room.stones)

                    if is_boulder:
                        if is_switch:
                            row_str += "[B+P]"
                        else:
                            row_str += "[ B ]"
                    elif is_stone:
                        row_str += "[ S ]"
                    elif is_switch:
                        row_str += "[ P ]"
                    elif tile == 1:
                        row_str += "[ W ]"
                    elif tile == 2:
                        row_str += "[ D ]"
                    elif tile == 3:
                        row_str += "[ E ]"
                    elif tile == 6:
                        row_str += "[KEY]"
                    elif tile == 9:
                        row_str += "[COL]"
                    else:
                        row_str += "[   ]"
                grid_viz.append(row_str)

            out.append("\n".join(grid_viz))
        out.append("\nCommand > ")

        return "\n".join(out)

    def _generate_hint(self):
        """
        Generate a context-aware hint based on the current puzzle type and state.
        """
        room = self.env.curr_room
        puzzle_type = room.puzzle_type.name

        hint_text = "\n--- HINT REQUESTED ---\n"

        if puzzle_type == "NONE":
            if room.is_exit:
                hint_text += "This room contains the EXIT! Step onto the center tile to escape the dungeon."
            elif room.has_key and self.last_obs['inventory'] == 0:
                ky, kx = room.key_pos
                lx, ly = self.last_obs['agent_pos']
                dx, dy = kx - lx, ky - ly
                dir_str = self._get_relative_dir(dx, dy)
                hint_text += f"There's a GOLDEN KEY in this room. It's located {dir_str}. Walk to it to pick it up."
            elif room.is_locked:
                hint_text += "This room's doors are locked. You need a key to proceed. Search other rooms for the golden key."
            else:
                hint_texts = ["This room is empty. Simply navigate to one of the doors to continue exploring.",
                              "Not much happening here, exit the room to continue exploring.",
                              "Have you tried all doors?",
                              "Try keeping systematic notes on room details to remind you where you've been and which doors you've tried."]
                hint_text += random.choice(hint_texts)

        elif puzzle_type == "BOULDER":
            hint_text += "BOULDER PUZZLE: Push boulders onto pressure plates to unlock the doors.\n"

            activated = sum(1 for s in room.switches if any(b[0] == s[0] and b[1] == s[1] for b in room.boulders))
            total = len(room.switches)

            hint_text += f"Progress: {activated}/{total} pressure plates activated.\n"

            if activated < total:
                hint_texts = [
                    "TIP: Walk into a boulder to push it one space in that direction.",
                    "TIP: Put the boulder between yourself and where you want to push it then walk into it.",
                    "TIP: Boulders can not be pushed up against a wall, they seem immobile if you try.",
                    "TIP: Try walking into the boulder from the other side if it seems immobile."
                ]
                hint_text += random.choice(hint_texts)
            else:
                hint_text += "All pressure plates are activated! The doors should be unlocked. Have you tried all doors?"

        elif puzzle_type == "STONE":
            hint_text += "SAILING STONE PUZZLE: Mystical stones move on their own across the floor.\n"
            hint_text += "TIP: Wait and observe their movement patterns. They push boulders and trigger pressure plates automatically. Sometimes patience is the solution."

        elif puzzle_type == "WARP_CYCLE":
            hint_text += "WARP CYCLE PUZZLE: Stepping on warp tiles will teleport you in a predictable pattern.\n"
            hint_text += "TIP: The teleportation follows a cycle. Experiment with the pattern to reach your destination. Some tiles might warp you closer to the exit or door."

        elif puzzle_type == "WARP_RND":
            hint_text += "RANDOM WARP PUZZLE: Chaos magic fills this room - warp tiles teleport you randomly.\n"
            hint_text += "TIP: There's no pattern here, just keep trying. Eventually you'll land where you need to be. Good luck!"

        else:
            hint_text += f"Unknown puzzle type: {puzzle_type}. Explore carefully and look for interactive elements."

        hint_text += "\n\nCommand > "
        return hint_text

    def _get_relative_dir(self, dx, dy):
        """
        Convert relative coordinate offset to natural language directional text.
        Handles 8-point compass directions with distance markers.
        """
        if dx == 0 and dy == 0:
            return "HERE"

        v_dir = ""
        if dy > 0:
            v_dir = "North"
        elif dy < 0:
            v_dir = "South"

        h_dir = ""
        if dx > 0:
            h_dir = "East"
        elif dx < 0:
            h_dir = "West"

        direction = f"{v_dir}-{h_dir}".strip("-")

        dist_str = []
        if dy != 0: dist_str.append(f"{abs(dy)}N" if dy > 0 else f"{abs(dy)}S")
        if dx != 0: dist_str.append(f"{abs(dx)}E" if dx > 0 else f"{abs(dx)}W")

        return f"{direction} (Steps: {', '.join(dist_str)})"