import gymnasium as gym
import numpy as np
import random


class NaturalLanguage(gym.Wrapper):
    """
    Wrapper that converts observations to natural language descriptions.

    Designed for LLM agents, this wrapper:
    - Replaces grid observations with rich text descriptions
    - Accepts text commands (NORTH, SOUTH, EAST, WEST, LOOK, RESET, EXIT)
    - Provides contextual flavor text and tactical information
    - Includes event logging for game state changes
    """
    def __init__(self, env):
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

        # Command mapping
        self.mapping = {
            'north': 0, 'n': 0, 'up': 0,
            'south': 1, 's': 1, 'down': 1,
            'east': 2, 'e': 2, 'right': 2,
            'west': 3, 'w': 3, 'left': 3
        }

        # Flavor Text Library
        self.flavor_texts = {
            "BOULDER": [
                "The floor is scarred with deep grooves. Heavy boulders sit silently, waiting to be moved.",
                "You hear the grinding of stone on stone. Pressure plates line the floor.",
                "A puzzle of weight and leverage lies before you."
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
            "INSTRUCTIONS: The dungeon is a grid. Solve puzzles to unlock doors. Find the Key. Reach the Exit.\n"
            "Commands: NORTH, SOUTH, EAST, WEST, LOOK, RESET, EXIT.\n"
        )
        return self._generate_text(obs, 0, False, intro_text), info

    def step(self, action_input: str):
        """
        Process a text command or numeric action and return text description.

        Accepted commands:
        - Directional: NORTH/N/UP, SOUTH/S/DOWN, EAST/E/RIGHT, WEST/W/LEFT
        - Utility: LOOK (re-describe room), RESET (new dungeon), EXIT/QUIT (terminate)

        Args:
            action_input: String command or integer action (0-3)

        Returns:
            tuple: (text description, reward, done, truncated, info)
        """
        # 1. PARSE
        if isinstance(action_input, str):
            cmd = action_input.lower().strip()

            if cmd in ['exit', 'quit']:
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

            events.append(f"You step through the door and enter a new room from the {entered_from}.")
            self.last_gx, self.last_gy = curr_gx, curr_gy
            self.last_solved = self.env.curr_room.check_solved()
        else:
            self.just_entered_room = False
            # Movement checks
            new_pos = obs['agent_pos']
            if not np.array_equal(prev_pos, new_pos):
                dist = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                if dist > 1:
                    events.append("WARP! The world twists—you reappear elsewhere!")
                else:
                    events.append("You move 1 step.")
            elif reward < 0:
                events.append("BLOCKED. You bump into a solid obstacle.")

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

        Args:
            room: Room instance

        Returns:
            str: Random flavor text appropriate to the room's puzzle type
        """
        p_type = room.puzzle_type.name
        if p_type not in self.flavor_texts: p_type = "NONE"
        return random.choice(self.flavor_texts[p_type])

    def _generate_text(self, obs, reward, done, event_text):
        """
        Generate complete text description of current game state.

        Includes:
        - Event log (what just happened)
        - Atmospheric narrative (on room entry)
        - Tactical situation report (coordinates, doors, objects)
        - 3x3 visual scan around player

        Args:
            obs: Observation dict from base environment
            reward: Current step reward
            done: Episode termination flag
            event_text: Description of recent events

        Returns:
            str: Complete multi-section text description
        """
        room = self.env.curr_room
        lx, ly = obs['agent_pos']
        w, h = room.w, room.h

        out = []

        # --- 1. EVENT LOG (Immediate Feedback) ---
        out.append(f"\n> {event_text}")

        # --- 2. NARRATIVE (On Entry) ---
        if self.just_entered_room:
            out.append(f"\n--- NEW CHAMBER DISCOVERED ---")
            out.append(f"\"{self._get_flavor_text(room)}\"")
            if room.is_locked and not room.has_puzzle:
                out.append("NOTICE: The door ahead is locked. You need a key.")

        # --- 3. TACTICAL OVERVIEW (Textual) ---
        out.append(f"\n--- SITUATION REPORT ---")
        out.append(f"LOCATION: Coordinate ({lx}, {ly}) in a {w}x{h} room.")

        # DOORS
        door_descs = []
        for d, off in room.doors.items():
            # Calculate distance/direction
            # N/S doors are at y=h-1 or y=0. E/W doors are at x=w-1 or x=0.
            # We can just say "North Wall" for clarity.
            state = "(OPEN)" if room.is_solved else "(LOCKED)"
            if room.puzzle_type.name == "NONE": state = "(OPEN)"
            door_descs.append(f"{d.name} Door at {d.name}-Wall: {state}")
        out.append("EXITS: " + ", ".join(door_descs))

        # INTERESTING OBJECTS (Relative to Player)
        # This is CRITICAL for LLMs that can't "see" the grid well
        objects = []

        # Switches
        for (sy, sx) in room.switches:
            covered = any(b[0] == sy and b[1] == sx for b in room.boulders)
            status = "ACTIVATED" if covered else "EMPTY"
            # Relative Dir
            dx, dy = sx - lx, sy - ly
            dir_str = self._get_relative_dir(dx, dy)
            objects.append(f"Pressure Plate ({status}): {dir_str}")

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
        out.append(f"\n--- VISUAL SCAN (3x3) ---")
        grid_viz = []
        for dy in [1, 0, -1]:
            row_str = ""
            for dx in [-1, 0, 1]:
                ny, nx = ly + dy, lx + dx

                if dx == 0 and dy == 0:
                    # Player
                    on_switch = (ny, nx) in room.switches
                    if on_switch:
                        row_str += "[YOU]"  # simplified
                    else:
                        row_str += "[YOU]"
                    continue

                if not (0 <= ny < h and 0 <= nx < w):
                    row_str += "[###]"
                    continue

                # Check contents
                tile = room.grid_array[ny, nx]
                is_boulder = any(b[0] == ny and b[1] == nx for b in room.boulders)
                is_switch = (ny, nx) in room.switches
                is_stone = any(s[0] == ny and s[1] == nx for s in room.stones)

                if is_boulder:
                    if is_switch:
                        row_str += "[OK!]"
                    else:
                        row_str += "[ @ ]"
                elif is_stone:
                    row_str += "[ * ]"
                elif is_switch:
                    row_str += "[ _ ]"
                elif tile == 1:
                    row_str += "[###]"
                elif tile == 2:
                    row_str += "[___]"
                elif tile == 3:
                    row_str += "[OUT]"
                elif tile == 6:
                    row_str += "[KEY]"
                else:
                    row_str += "  .  "
            grid_viz.append(row_str)

        out.append("\n".join(grid_viz))
        out.append("\nCommand > ")

        return "\n".join(out)

    def _get_relative_dir(self, dx, dy):
        """
        Convert relative coordinate offset to directional text.

        Args:
            dx: X offset from player position
            dy: Y offset from player position

        Returns:
            str: Direction text like "2 North, 1 East" or "HERE"
        """
        if dx == 0 and dy == 0: return "HERE"
        parts = []
        if dy > 0:
            parts.append(f"{dy} North")
        elif dy < 0:
            parts.append(f"{abs(dy)} South")

        if dx > 0:
            parts.append(f"{dx} East")
        elif dx < 0:
            parts.append(f"{abs(dx)} West")

        return ", ".join(parts)