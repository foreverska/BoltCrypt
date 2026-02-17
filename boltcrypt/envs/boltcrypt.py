import cv2
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces

from boltcrypt.envs.defines import *
from boltcrypt.envs.dungeon_generator import DungeonGenerator

class BoltCrypt(gym.Env):
    """
    OpenAI Gymnasium environment for procedurally generated dungeon exploration.

    Features:
    - Procedural room-based dungeons with variable connectivity
    - Multiple puzzle types (boulder pushing, pressure plates, teleporters, moving obstacles)
    - Key-lock mechanics requiring exploration
    - Local room observations with global position tracking
    - Inventory system for keys
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(self, render_mode=None, generator_config=None, puzzle_bonus=True, key_bonus=True):
        """
        Initialize the BoltCrypt environment.

        Args:
            render_mode: The rendering mode to use, can be "human" or "rgb_array"
            generator_config: Dict of parameters for DungeonGenerator (see DungeonGenerator.__init__)
            puzzle_bonus: If True, grant +1.0 reward for solving boulder puzzles
            key_bonus: If True, grant +1.0 reward for collecting the key
        """
        super().__init__()
        self.render_mode = render_mode
        self.gen_config = generator_config or {}
        self.generator = DungeonGenerator(**self.gen_config)
        self.action_space = spaces.Discrete(4)
        self.puzzle_bonus = puzzle_bonus
        self.key_bonus = key_bonus

        self.gx, self.gy = None, None
        self.curr_room = None
        self.lx = None
        self.ly = None
        self.episode_step = None
        self.inventory_has_key = None

        # Obs: Grid + AgentPos + GlobalPos + Inventory
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=10, shape=(self.generator.max_room_dim, self.generator.max_room_dim), dtype=np.int8),
            "agent_pos": spaces.Box(low=0, high=self.generator.max_room_dim, shape=(2,), dtype=np.int32),
            "global_pos": spaces.Box(low=-1000, high=1000, shape=(2,), dtype=np.int32),
            "inventory": spaces.Discrete(2)
        })

    def reset(self, seed=None, options=None):
        """
        Reset the environment and generate a new dungeon.

        Args:
            seed: Random seed for dungeon generation
            options: Additional options (currently unused)

        Returns:
            tuple: (observation dict, info dict)
        """
        super().reset(seed=seed)
        self.generator.seed(seed)
        self.generator.generate()

        self.gx, self.gy = self.generator.start_pos
        self.curr_room = self.generator.grid[(self.gx, self.gy)]
        self.lx = self.curr_room.w // 2
        self.ly = self.curr_room.h // 2
        self.episode_step = 0
        self.inventory_has_key = False

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one environment step with the given action.

        Actions: 0=North, 1=South, 2=East, 3=West

        Args:
            action: Integer action (0-3)

        Returns:
            tuple: (observation, reward, done, truncated, info)
                - observation: Dict with 'grid', 'agent_pos', 'global_pos', 'inventory'
                - reward: Float reward (-0.01 per step, +1.0 for key/puzzle, +10.0 for exit)
                - done: Bool indicating episode termination (reaching exit)
                - truncated: Bool (always False)
                - info: Dict (empty)
        """
        # 0. PRE-MOVE: Sailing Stones Move
        self.curr_room.move_stones(self.generator.rng, (self.ly, self.lx))

        # 1. PARSE ACTION
        dx, dy, direction = 0, 0, None
        if action == 0:
            dx, dy, direction = 0, 1, Direction.NORTH
        elif action == 1:
            dx, dy, direction = 0, -1, Direction.SOUTH
        elif action == 2:
            dx, dy, direction = 1, 0, Direction.EAST
        elif action == 3:
            dx, dy, direction = -1, 0, Direction.WEST

        target_lx, target_ly = self.lx + dx, self.ly + dy
        reward = -0.01
        done = False

        # 2. IN-ROOM MOVEMENT
        if 0 <= target_lx < self.curr_room.w and 0 <= target_ly < self.curr_room.h:
            static_tile = self.curr_room.grid_array[target_ly, target_lx]

            # Check Sailing Stone Collision (Treat as Wall)
            hit_stone = any(s[0] == target_ly and s[1] == target_lx for s in self.curr_room.stones)

            if hit_stone:
                pass  # Blocked

            # Check Key
            elif static_tile == TILE_KEY:
                self.inventory_has_key = True
                self.curr_room.grid_array[target_ly, target_lx] = TILE_EMPTY
                if self.key_bonus: reward = 1.0
                self.lx, self.ly = target_lx, target_ly

            # Check Boulder Push
            elif any(b[0] == target_ly and b[1] == target_lx for b in self.curr_room.boulders):
                b_idx = next(
                    i for i, b in enumerate(self.curr_room.boulders) if b[0] == target_ly and b[1] == target_lx)

                # --- STANDARD BOULDER MECHANIC ---
                push_lx, push_ly = target_lx + dx, target_ly + dy
                is_safe = (2 <= push_lx <= self.curr_room.w - 3 and 2 <= push_ly <= self.curr_room.h - 3)

                if is_safe:
                    push_tile = self.curr_room.grid_array[push_ly, push_lx]
                    is_blocked_push = push_tile in [TILE_WALL, TILE_COLUMN, TILE_DOOR, TILE_EXIT]
                    if any(b[0] == push_ly and b[1] == push_lx for b in
                           self.curr_room.boulders): is_blocked_push = True
                    if any(
                        s[0] == push_ly and s[1] == push_lx for s in self.curr_room.stones): is_blocked_push = True

                    if not is_blocked_push:
                        self.curr_room.boulders[b_idx] = [push_ly, push_lx]
                        self.lx, self.ly = target_lx, target_ly

                        previously_solved = getattr(self.curr_room, 'is_solved', False)
                        solved = self.curr_room.check_solved()
                        if not previously_solved and solved and self.puzzle_bonus:
                            reward = 1.0

            # Standard Movement
            elif static_tile != TILE_WALL and static_tile != TILE_COLUMN:
                # Door Transition Check
                if static_tile == TILE_DOOR:
                    # Check if door is locked
                    is_locked = False
                    if self.curr_room.puzzle_type in [PuzzleType.BOULDER] and not getattr(
                            self.curr_room, 'is_solved', True):
                        is_locked = True
                    elif getattr(self.curr_room, 'puzzle_type', None) == PuzzleType.BOULDER_PLATES:
                        if getattr(self.curr_room, 'active_plate', None) is None:
                            is_locked = True
                        else:
                            unlocked_door = self.curr_room.plate_door_map.get(self.curr_room.active_plate)
                            if unlocked_door != direction:
                                is_locked = True

                    # If unlocked, trigger room transition
                    if not is_locked and direction and direction in getattr(self.curr_room, 'doors', {}):
                        next_gx = self.gx + direction.value[0]
                        next_gy = self.gy + direction.value[1]
                        target_room = self.generator.grid.get((next_gx, next_gy))

                        if target_room:
                            # Check target room entry lock (Key)
                            if not (getattr(target_room, 'is_locked', False) and not getattr(self, 'inventory_has_key', False)):
                                self.gx, self.gy = next_gx, next_gy
                                self.curr_room = target_room
                                off = self.curr_room.doors.get(direction.opposite, 0)

                                # Place player one tile in front of the door
                                if direction == Direction.NORTH:
                                    self.lx, self.ly = off, 1
                                elif direction == Direction.SOUTH:
                                    self.lx, self.ly = off, self.curr_room.h - 2
                                elif direction == Direction.EAST:
                                    self.lx, self.ly = 1, off
                                elif direction == Direction.WEST:
                                    self.lx, self.ly = self.curr_room.w - 2, off

                # Normal tile movement (not door)
                elif static_tile != TILE_DOOR:
                    self.lx, self.ly = target_lx, target_ly
                    if static_tile == TILE_EXIT:
                        reward = 10.0
                        done = True

        # 3. OUT-OF-BOUNDS (Should not happen with proper door placement, but kept for safety)
        # Room transitions are now handled when stepping onto door tiles in section 2

        # 4. POST-MOVE: Warp Check
        if hasattr(self.curr_room, 'warps') and (self.ly, self.lx) in self.curr_room.warps:
            warp_info = self.curr_room.warps[(self.ly, self.lx)]

            dest = None
            if getattr(self.curr_room, 'warp_mode', None) == PuzzleType.WARP_CYCLE:
                dest = warp_info
            elif getattr(self.curr_room, 'warp_mode', None) == PuzzleType.WARP_RND:
                opts = [w for w in warp_info if w != (self.ly, self.lx)]
                if opts: dest = self.generator.rng.choice(opts)

            if dest:
                stone_at_dest = any(s[0] == dest[0] and s[1] == dest[1] for s in getattr(self.curr_room, 'stones', []))
                if not stone_at_dest:
                    self.ly, self.lx = dest

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create a base grid
        full_grid = np.full((self.generator.max_room_dim, self.generator.max_room_dim), TILE_WALL, dtype=np.int8)
        h, w = self.curr_room.h, self.curr_room.w
        copy_h, copy_w = min(h, self.generator.max_room_dim), min(w, self.generator.max_room_dim)

        room_grid = self.curr_room.grid_array[:copy_h, :copy_w].copy()

        # Overlay Columns
        for c in self.curr_room.columns:
            room_grid[c[0], c[1]] = TILE_COLUMN

        # Overlay Dynamic Entities
        # 1. Boulders
        for b in self.curr_room.boulders:
            if b[0] < copy_h and b[1] < copy_w:
                    room_grid[b[0], b[1]] = TILE_BOULDER

        # 2. Sailing Stones
        for s in self.curr_room.stones:
            if s[0] < copy_h and s[1] < copy_w:
                room_grid[s[0], s[1]] = TILE_STONE

        full_grid[:copy_h, :copy_w] = room_grid

        return {
            "grid": full_grid,
            "agent_pos": np.array([self.lx, self.ly], dtype=np.int32),
            "global_pos": np.array([self.gx, self.gy], dtype=np.int32),
            "inventory": 1 if self.inventory_has_key else 0
        }

    def obs_to_rgb(self):
        # Create a palette: shape (max_tile_id + 1, 3)
        palette = np.array([
            [230, 215, 180],  # 0: Floor
            [40, 40, 40],  # 1: Wall
            [139, 69, 19],  # 2: Door
            [0, 255, 0],  # 3: Exit
            [200, 50, 50],  # 4: Switch
            [100, 80, 50],  # 5: Boulder
            [255, 215, 0],  # 6: Key
            [100, 100, 100],  # 7: Stone
            [148, 0, 211],  # 8: Warp
        ], dtype=np.uint8)

        # Map the grid to RGB
        obs = self._get_obs()
        grid = obs['grid']
        img = palette[grid]  # Magic of NumPy indexing

        # Overlay Agent (since agent isn't in the grid)
        lx, ly = obs['agent_pos']
        img[ly, lx] = [50, 100, 200]

        return img

    def render(self):
        if self.render_mode == "rgb":
            return self.obs_to_rgb()
        elif self.render_mode == "human":
            img = self.obs_to_rgb()

            # Upscale 10x10 to 320x320 for visibility
            display_img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)

            # Convert RGB to BGR for OpenCV
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

            cv2.imshow("BoltCrypt", display_img)
            cv2.waitKey(1)  # Refresh window

        return None