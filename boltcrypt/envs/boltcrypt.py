import cv2
import numpy as np
import random
from enum import Enum
import gymnasium as gym
import gymnasium.spaces as spaces

# --- Constants ---
TILE_EMPTY = 0
TILE_WALL = 1
TILE_DOOR = 2
TILE_EXIT = 3
TILE_SWITCH = 4
TILE_BOULDER = 5
TILE_KEY = 6
TILE_STONE = 7  # RND "Sailing Stone"
TILE_WARP = 8  # ICM "Warp Tile"

# DIMENSIONS
MIN_ROOM_DIM = 4  # Bumped up slightly to fit puzzles
MAX_ROOM_DIM = 10


class Direction(Enum):
    NORTH = (0, 1)
    SOUTH = (0, -1)
    EAST = (1, 0)
    WEST = (-1, 0)

    @property
    def opposite(self):
        return _OPPOSITE_LOOKUP[self]


_OPPOSITE_LOOKUP = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST
}


class PuzzleType(Enum):
    NONE = "none"
    BOULDER = "boulder"  # Push blocks to switches
    BOULDER_PLATES = "mapped_plates"  # One boulder, multiple pressure plates
    STONE = "stone"  # Randomly moving obstacles (RND Killer)
    WARP_CYCLE = "warp_cycle"  # Hamilton cycle teleporters (ICM Killer)
    WARP_RND = "warp_rnd"  # Degenerate random teleporters (ICM Killer++)


class Room:
    """
    Represents a single room in the dungeon with puzzle mechanics.

    Rooms can contain various puzzle types (boulders, pressure plates, teleporters, moving obstacles)
    and may have doors connecting to adjacent rooms.
    """
    __slots__ = ['x', 'y', 'w', 'h', 'is_start', 'is_exit', 'is_locked', 'has_key',
                 'doors', '_cached_grid', 'switches', 'boulders', 'stones', 'warps',
                 'warp_mode', 'puzzle_type', 'is_solved', 'previously_solved', 'key_pos',
                 'active_plate', 'plate_door_map']

    def __init__(self, x: int, y: int, w: int, h: int, is_start=False, is_exit=False):
        """
        Initialize a room at dungeon coordinates (x, y) with dimensions (w, h).

        Args:
            x: Global x-coordinate in dungeon grid
            y: Global y-coordinate in dungeon grid
            w: Room width
            h: Room height
            is_start: Whether this is the starting room
            is_exit: Whether this is the exit room
        """
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.is_start = is_start
        self.is_exit = is_exit
        self.is_locked = False
        self.has_key = False
        self.key_pos = None

        self.doors = {}
        self.puzzle_type = PuzzleType.NONE
        self.is_solved = True
        self.previously_solved = False

        # Puzzle Specific Data
        self.switches = set()
        self.boulders = []
        self.stones = []  # List of [y, x] for sailing stones
        self.warps = {}  # Map (y,x) -> (y,x) or list of locs
        self.warp_mode = None
        self.active_plate = None  # For boulder_plates: which plate is currently occupied
        self.plate_door_map = {}  # For boulder_plates: maps plate pos -> door direction

        self._cached_grid = None

    def add_door(self, direction, offset):
        """
        Add a door in the specified direction at the given offset along the wall.

        Args:
            direction: Direction enum value (NORTH, SOUTH, EAST, WEST)
            offset: Position along the wall where the door is placed
        """
        self.doors[direction] = offset

    def setup_puzzle(self, rng, puzzle_density: float, allowed_types: list):
        """
        Generate and configure a puzzle for this room.

        Puzzles include: boulder pushing, pressure plates, sailing stones, and warp tiles.
        Rooms marked as start, exit, or containing keys don't receive puzzles.

        Args:
            rng: Random number generator instance
            puzzle_density: Probability (0-1) of generating a puzzle in this room
            allowed_types: List of allowed puzzle type strings (e.g., ["boulder", "stone"])
        """
        if self.is_start or self.is_exit or self.has_key: return
        if rng.random() > puzzle_density: return
        if not allowed_types: return

        # Pick a puzzle type for this room
        p_type_str = rng.choice(allowed_types)
        self.puzzle_type = PuzzleType(p_type_str)

        # Get valid interior coordinates
        candidates = []
        for y in range(2, self.h - 2):
            for x in range(2, self.w - 2):
                candidates.append((y, x))

        if len(candidates) < 2:
            self.puzzle_type = PuzzleType.NONE
            return

        # --- GENERATE PUZZLE ---

        if self.puzzle_type == PuzzleType.BOULDER:
            max_elements = len(candidates) // 2
            num_elements = rng.randint(1, max(1, min(3, max_elements)))
            chosen = rng.sample(candidates, num_elements * 2)
            for i in range(num_elements):
                self.switches.add(chosen[i])
                self.boulders.append(list(chosen[i + num_elements]))
            self.is_solved = False

        elif self.puzzle_type == PuzzleType.BOULDER_PLATES:
            # One boulder, multiple pressure plates (one per door)
            num_plates = len(self.doors)
            if num_plates == 0 or len(candidates) < num_plates + 1:
                self.puzzle_type = PuzzleType.NONE
                return

            # Place pressure plates and boulder
            chosen = rng.sample(candidates, num_plates + 1)
            for i in range(num_plates):
                self.switches.add(chosen[i])
            self.boulders.append(list(chosen[num_plates]))

            # Randomly map each pressure plate to a door
            door_list = list(self.doors.keys())
            rng.shuffle(door_list)
            for i, plate_pos in enumerate(chosen[:num_plates]):
                self.plate_door_map[plate_pos] = door_list[i]

            # Check initial state
            boulder_pos = tuple(self.boulders[0])
            if boulder_pos in self.plate_door_map:
                self.active_plate = boulder_pos
            self.is_solved = False

        elif self.puzzle_type == PuzzleType.STONE:
            # RND Sailing Stones
            num_stones = rng.randint(1, 3)
            # Ensure we don't block everything, pick sparse locations
            chosen = rng.sample(candidates, min(num_stones, len(candidates)))
            self.stones = [list(pos) for pos in chosen]
            # Stones don't need "solving", they are just hazards/noise
            self.is_solved = True

        elif self.puzzle_type in [PuzzleType.WARP_CYCLE, PuzzleType.WARP_RND]:
            # ICM Warp Tiles
            # 1. Gather ALL valid floor tiles (no safety buffer)
            candidates = []
            for y in range(1, self.h - 1):
                for x in range(1, self.w - 1):
                    candidates.append((y, x))
            if len(candidates) < 2:
                self.puzzle_type = PuzzleType.NONE
                return
            # 2. Shuffle is CRITICAL for the Hamilton Cycle
            # If we don't shuffle, the cycle might just be linear (1,1)->(1,2)->(1,3)
            # which would just act like a conveyor belt. We want chaos.
            rng.shuffle(candidates)
            self.warp_mode = self.puzzle_type
            if self.puzzle_type == PuzzleType.WARP_CYCLE:
                # Hamilton Cycle: Create a closed loop of ALL floor tiles.
                # Stepping on candidates[i] warps you to candidates[i+1].
                # To traverse the room, the agent must learn the specific
                # permutation of the room's topology.
                for i in range(len(candidates)):
                    src = candidates[i]
                    dst = candidates[(i + 1) % len(candidates)]
                    self.warps[src] = dst
            else:
                # Degenerate (Total Chaos):
                # Every tile maps to the list of ALL tiles.
                # Stepping anywhere warps you somewhere random.
                # This breaks the "Next State Prediction" entirely.
                for pos in candidates:
                    self.warps[pos] = candidates

            self.is_solved = True

    def place_key(self, rng):
        """
        Place a key at a random valid position in the room.

        Args:
            rng: Random number generator instance
        """
        self.has_key = True
        candidates = []
        for y in range(1, self.h - 1):
            for x in range(1, self.w - 1):
                    candidates.append((y, x))
        if candidates:
            self.key_pos = rng.choice(candidates)

    def move_stones(self, rng, agent_pos):
        """
        Move all sailing stones one step randomly (for RND puzzle type).

        Stones attempt random movements but cannot move into walls, doors, boulders,
        other stones, or the agent's position.

        Args:
            rng: Random number generator instance
            agent_pos: Current (y, x) position of the agent to avoid collisions
        """
        if not self.stones: return

        for i in range(len(self.stones)):
            sy, sx = self.stones[i]

            # Try random directions
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            rng.shuffle(moves)

            for dy, dx in moves:
                ny, nx = sy + dy, sx + dx

                # Check Bounds & Walls
                if not (1 <= ny < self.h - 1 and 1 <= nx < self.w - 1): continue
                if self.grid_array[ny, nx] in [TILE_WALL, TILE_DOOR, TILE_EXIT]: continue

                # Check Agent Collision (Stones don't push agent)
                if (ny, nx) == agent_pos: continue

                # Check Other Stones
                collision = False
                for j, other in enumerate(self.stones):
                    if i != j and other[0] == ny and other[1] == nx:
                        collision = True
                        break
                if collision: continue

                # Check Boulders (Stones don't push boulders)
                if any(b[0] == ny and b[1] == nx for b in self.boulders): continue

                # Success
                self.stones[i] = [ny, nx]
                break

    def freeze(self):
        """
        Generate and cache the static grid background for this room.

        Creates the base tile layout including walls, floors, doors, switches, warps,
        exit tile, and key. Dynamic entities (boulders, stones) are added at render time.
        """
        grid = np.full((self.h, self.w), TILE_WALL, dtype=np.int8)
        grid[1:-1, 1:-1] = TILE_EMPTY

        # Doors
        for direction, offset in self.doors.items():
            if direction == Direction.NORTH:
                grid[-1, offset] = TILE_DOOR
            elif direction == Direction.SOUTH:
                grid[0, offset] = TILE_DOOR
            elif direction == Direction.EAST:
                grid[offset, -1] = TILE_DOOR
            elif direction == Direction.WEST:
                grid[offset, 0] = TILE_DOOR

        # Switches
        for (sy, sx) in self.switches:
            grid[sy, sx] = TILE_SWITCH

        # Warps (Static positions)
        for (wy, wx) in self.warps.keys():
            grid[wy, wx] = TILE_WARP

        # Exit/Key
        if self.is_exit:
            grid[self.h // 2, self.w // 2] = TILE_EXIT
        if self.has_key and self.key_pos:
            grid[self.key_pos[0], self.key_pos[1]] = TILE_KEY

        self._cached_grid = grid

    def check_solved(self):
        """
        Check if the room's puzzle is solved and update puzzle state.

        Returns:
            bool: True if puzzle is solved or has no puzzle, False otherwise
        """
        if self.puzzle_type == PuzzleType.BOULDER:
            boulder_tuples = {tuple(b) for b in self.boulders}
            self.is_solved = self.switches.issubset(boulder_tuples)
            if self.is_solved and self.previously_solved == False:
                self.previously_solved = True
            return self.is_solved
        elif self.puzzle_type == PuzzleType.BOULDER_PLATES:
            # Update active plate based on boulder position
            boulder_pos = tuple(self.boulders[0])
            if boulder_pos in self.plate_door_map:
                self.active_plate = boulder_pos
            else:
                self.active_plate = None
            # Puzzle is never fully "solved", just changes state
            self.is_solved = False
            return False
        else:
            self.is_solved = True
            return True

    @property
    def grid_array(self):
        return self._cached_grid


class DungeonGenerator:
    """
    Procedural dungeon generator creating a connected graph of rooms with puzzles.

    Generates dungeons by:
    1. Building a skeleton path from start to a minimum distance
    2. Growing additional rooms to reach target room count
    3. Selecting an exit room from valid vault candidates
    4. Adding extra connectivity (loops)
    5. Placing puzzles and optional key-lock mechanisms
    """
    def __init__(self, seed: int = None, min_dist=5, max_dist=None, mean_rooms=15, std_rooms=2, connectivity=0.5,
                 puzzle_density=0.0, allowed_puzzles=None, key_puzzle_prob=0.0, puzzle_required=False,
                 min_room_dim=MIN_ROOM_DIM, max_room_dim=MAX_ROOM_DIM):
        """
        Initialize the dungeon generator with configuration parameters.

        Args:
            seed: Random seed for reproducibility
            min_dist: Minimum Manhattan distance from start to exit
            max_dist: Maximum Manhattan distance for exit selection (None = use min_dist)
            mean_rooms: Average number of rooms to generate
            std_rooms: Standard deviation for room count
            connectivity: Probability (0-1) of adding extra connections (loops)
            puzzle_density: Probability (0-1) of placing puzzles in rooms
            allowed_puzzles: List of allowed puzzle types (defaults to all)
            key_puzzle_prob: Probability (0-1) of requiring a key to unlock exit
            puzzle_required: If True, guarantees a puzzle in the room leading to exit vault (overrides puzzle_density for that room)
            min_room_dim: Minimum room dimension
            max_room_dim: Maximum room dimension
        """
        self.seed(seed)
        self.config = {
            'min_dist': min_dist, 'max_dist': max_dist, 'mean_rooms': mean_rooms,
            'std_rooms': std_rooms, 'connectivity': connectivity,
            'puzzle_density': puzzle_density, 'key_puzzle_prob': key_puzzle_prob,
            'puzzle_required': puzzle_required
        }
        # Default to all puzzles if not specified
        self.allowed_puzzles = allowed_puzzles if allowed_puzzles else ["boulder", "mapped_plates", "stone", "warp_cycle", "warp_rnd"]

        self.grid = {}
        self.start_pos = (0, 0)
        self.end_pos = None
        self.skeleton_end_pos = None  # Where skeleton path ends
        self.min_room_dim = min_room_dim
        self.max_room_dim = max_room_dim

    def seed(self, seed=None):
        """
        Set the random seed for dungeon generation.

        Args:
            seed: Random seed value (None for random seed)
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(self):
        """
        Generate a complete dungeon layout with rooms, doors, puzzles, and objectives.

        This is the main generation pipeline that orchestrates all dungeon creation steps.
        """
        self.grid.clear()
        target_count = int(self.np_rng.normal(self.config['mean_rooms'], self.config['std_rooms']))

        # Ensure we have enough rooms for the distance requirement
        min_dist = self.config['min_dist']
        max_dist = self.config['max_dist']
        required_min = max_dist if max_dist is not None else min_dist
        target_count = max(target_count, required_min + 2)

        self._build_skeleton()
        self._grow_to_target(target_count)
        self._select_goal_room()
        self._apply_connectivity()
        self._apply_key_puzzle()

        # Find the room leading to the exit vault (if puzzle_required is enabled)
        vault_parent_room = None
        for pos, r in self.grid.items():
            for d in Direction:
                neighbor = (pos[0] + d.value[0], pos[1] + d.value[1])
                if neighbor == self.end_pos:
                    vault_parent_room = r
                    break

        # Apply puzzles to all rooms
        for pos, r in self.grid.items():
            # If this is the vault parent and puzzle_required is on, force puzzle density to 1.0
            if vault_parent_room is not None and r is vault_parent_room:
                r.setup_puzzle(self.rng, 1.0, self.allowed_puzzles)
            else:
                r.setup_puzzle(self.rng, self.config['puzzle_density'], self.allowed_puzzles)
            r.freeze()

    def _connect_rooms(self, r1, r2, direction):
        if direction in r1.doors: return
        limit = min(r1.w, r2.w) if direction in [Direction.NORTH, Direction.SOUTH] else min(r1.h, r2.h)
        if limit < 3: return
        offset = self.rng.randint(1, limit - 2)
        r1.add_door(direction, offset)
        r2.add_door(direction.opposite, offset)

    def _get_random_dims(self):
        w = self.rng.randint(self.min_room_dim, self.max_room_dim)
        h = self.rng.randint(self.min_room_dim, self.max_room_dim)
        return w, h

    def _build_skeleton(self):
        self.start_pos = (0, 0)
        # Build skeleton to max_dist if specified, else min_dist
        skeleton_dist = self.config['max_dist'] if self.config['max_dist'] is not None else self.config['min_dist']

        dist_x = self.rng.randint(0, skeleton_dist)
        dist_y = skeleton_dist - dist_x
        dx = dist_x * self.rng.choice([1, -1])
        dy = dist_y * self.rng.choice([1, -1])
        self.skeleton_end_pos = (dx, dy)

        current = self.start_pos
        path_coords = [current]
        while current != self.skeleton_end_pos:
            cx, cy = current
            ex, ey = self.skeleton_end_pos
            candidates = []
            if ex > cx: candidates.append(Direction.EAST)
            if ex < cx: candidates.append(Direction.WEST)
            if ey > cy: candidates.append(Direction.NORTH)
            if ey < cy: candidates.append(Direction.SOUTH)
            move = self.rng.choice(candidates)
            current = (cx + move.value[0], cy + move.value[1])
            path_coords.append(current)

        # Create rooms but don't mark any as exit yet
        for i, pos in enumerate(path_coords):
            is_start = (pos == self.start_pos)
            w, h = self._get_random_dims()
            self.grid[pos] = Room(pos[0], pos[1], w, h, is_start, is_exit=False)
            if i > 0:
                prev = path_coords[i - 1]
                dx, dy = pos[0] - prev[0], pos[1] - prev[1]
                self._connect_rooms(self.grid[prev], self.grid[pos], Direction((dx, dy)))

    def _grow_to_target(self, target_count):
        while len(self.grid) < target_count:
            candidates = []
            for pos in self.grid:
                for d in Direction:
                    neighbor = (pos[0] + d.value[0], pos[1] + d.value[1])
                    if neighbor not in self.grid:
                        candidates.append((neighbor, pos, d))
            if not candidates: break
            new_pos, parent, d = self.rng.choice(candidates)
            w, h = self._get_random_dims()
            new_room = Room(new_pos[0], new_pos[1], w, h)
            self.grid[new_pos] = new_room
            self._connect_rooms(self.grid[parent], new_room, d)

    def _select_goal_room(self):
        """
        Select the exit room from vault rooms (1-door rooms) within the distance range.

        Prioritizes dead-end rooms at valid distances (min_dist to max_dist) from start.
        Falls back to skeleton end position if no valid vaults exist.
        """
        min_dist = self.config['min_dist']
        max_dist = self.config['max_dist']

        # Calculate Manhattan distance from start for all rooms
        def manhattan_dist(pos):
            return abs(pos[0] - self.start_pos[0]) + abs(pos[1] - self.start_pos[1])

        # Find all vaults (rooms with exactly 1 door) in the valid distance range
        vaults = []
        for pos, room in self.grid.items():
            if pos == self.start_pos:
                continue
            dist = manhattan_dist(pos)
            if len(room.doors) == 1:
                # Check distance constraints
                if max_dist is None:
                    # If no max_dist, just use rooms at min_dist or greater
                    if dist >= min_dist:
                        vaults.append(pos)
                else:
                    # If max_dist specified, use range [min_dist, max_dist]
                    if min_dist <= dist <= max_dist:
                        vaults.append(pos)

        # If we found vaults in range, pick one randomly
        if vaults:
            self.end_pos = self.rng.choice(vaults)
        else:
            # Fallback: pick skeleton end if no vaults found
            self.end_pos = self.skeleton_end_pos

        # Mark the selected room as exit
        self.grid[self.end_pos].is_exit = True

    def _apply_connectivity(self):
        if self.config['connectivity'] <= 0: return
        links = []
        for pos, room in self.grid.items():
            for d in Direction:
                n_pos = (pos[0] + d.value[0], pos[1] + d.value[1])
                if n_pos in self.grid and d not in room.doors:
                    neighbor = self.grid[n_pos]
                    if pos < n_pos:
                        links.append((room, neighbor, d))
        self.rng.shuffle(links)
        for i in range(int(len(links) * self.config['connectivity'])):
            self._connect_rooms(*links[i])

    def _apply_key_puzzle(self):
        """
        Randomly apply a key-lock puzzle to the dungeon.

        With probability key_puzzle_prob, locks the exit room and places a key
        in a dead-end room (or random inner room if no dead-ends exist).
        """
        if self.rng.random() > self.config['key_puzzle_prob']: return

        # 1. Lock the Exit Room
        if self.end_pos in self.grid:
            self.grid[self.end_pos].is_locked = True

        # 2. Find Candidates
        # Exclude Start (0,0) and Exit (end_pos) from holding the key
        inner_rooms = [r for pos, r in self.grid.items() if pos not in [self.start_pos, self.end_pos]]

        if not inner_rooms:
            return

        # 3. Prioritize Dead Ends (Rooms with only 1 door)
        dead_ends = [r for r in inner_rooms if len(r.doors) == 1]

        if dead_ends:
            # Primary: Hide key in a dead end to force exploration
            self.rng.choice(dead_ends).place_key(self.rng)
        else:
            # Fallback: No dead ends exist (rare, but possible in loops), pick random inner room
            self.rng.choice(inner_rooms).place_key(self.rng)


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

        # Obs: Grid + AgentPos + GlobalPos + Inventory
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=9, shape=(self.generator.max_room_dim, self.generator.max_room_dim), dtype=np.int8),
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
        # (Pass current agent pos to avoid stones crushing player)
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
            hit_stone = False
            for s in self.curr_room.stones:
                if s[0] == target_ly and s[1] == target_lx:
                    hit_stone = True
                    break

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
                b_idx = -1
                for i, b in enumerate(self.curr_room.boulders):
                    if b[0] == target_ly and b[1] == target_lx: b_idx = i

                push_lx, push_ly = target_lx + dx, target_ly + dy
                is_safe = (2 <= push_lx <= self.curr_room.w - 3 and 2 <= push_ly <= self.curr_room.h - 3)

                if is_safe:
                    push_tile = self.curr_room.grid_array[push_ly, push_lx]
                    # Boulders can't push into walls, doors, exits, OR other boulders, OR stones
                    is_blocked_push = False
                    if push_tile in [TILE_WALL, TILE_DOOR, TILE_EXIT]: is_blocked_push = True
                    if any(b[0] == push_ly and b[1] == push_lx for b in self.curr_room.boulders): is_blocked_push = True
                    if any(s[0] == push_ly and s[1] == push_lx for s in self.curr_room.stones): is_blocked_push = True

                    if not is_blocked_push:
                        self.curr_room.boulders[b_idx] = [push_ly, push_lx]
                        self.lx, self.ly = target_lx, target_ly
                        previously_solved = self.curr_room.previously_solved
                        solved = self.curr_room.check_solved()
                        if previously_solved == False and solved and self.puzzle_bonus:
                            reward = 1.0

            # Standard Movement
            elif static_tile == TILE_DOOR:
                # Puzzle Lock Check
                is_locked = False
                if self.curr_room.puzzle_type == PuzzleType.BOULDER and not self.curr_room.is_solved:
                    is_locked = True
                elif self.curr_room.puzzle_type == PuzzleType.BOULDER_PLATES:
                    # Check if this door is unlocked by the current active plate
                    if self.curr_room.active_plate is None:
                        is_locked = True
                    else:
                        unlocked_door = self.curr_room.plate_door_map.get(self.curr_room.active_plate)
                        if unlocked_door != direction:
                            is_locked = True

                if not is_locked:
                    self.lx, self.ly = target_lx, target_ly

            elif static_tile != TILE_WALL:
                self.lx, self.ly = target_lx, target_ly
                if static_tile == TILE_EXIT:
                    reward = 10.0
                    done = True

        # 3. ROOM TRANSITION
        else:
            if direction and direction in self.curr_room.doors:
                next_gx = self.gx + direction.value[0]
                next_gy = self.gy + direction.value[1]
                target_room = self.generator.grid[(next_gx, next_gy)]

                is_blocked = False
                # Room Exit Lock (Boulders)
                if self.curr_room.puzzle_type == PuzzleType.BOULDER and not self.curr_room.is_solved:
                    is_blocked = True
                # Room Exit Lock (Boulder Plates)
                elif self.curr_room.puzzle_type == PuzzleType.BOULDER_PLATES:
                    if self.curr_room.active_plate is None:
                        is_blocked = True
                    else:
                        unlocked_door = self.curr_room.plate_door_map.get(self.curr_room.active_plate)
                        if unlocked_door != direction:
                            is_blocked = True
                # Target Room Entry Lock (Key)
                if target_room.is_locked and not self.inventory_has_key:
                    is_blocked = True

                if not is_blocked:
                    self.gx, self.gy = next_gx, next_gy
                    self.curr_room = target_room
                    off = self.curr_room.doors[direction.opposite]
                    if direction == Direction.NORTH:
                        self.lx, self.ly = off, 0
                    elif direction == Direction.SOUTH:
                        self.lx, self.ly = off, self.curr_room.h - 1
                    elif direction == Direction.EAST:
                        self.lx, self.ly = 0, off
                    elif direction == Direction.WEST:
                        self.lx, self.ly = self.curr_room.w - 1, off

        # 4. POST-MOVE: Warp Check
        # If we landed on a warp tile, teleport immediately
        if (self.ly, self.lx) in self.curr_room.warps:
            warp_info = self.curr_room.warps[(self.ly, self.lx)]

            dest = None
            if self.curr_room.warp_mode == PuzzleType.WARP_CYCLE:
                dest = warp_info  # It's a tuple (y,x)
            elif self.curr_room.warp_mode == PuzzleType.WARP_RND:
                # It's a list of all warp tiles, pick random distinct one
                opts = [w for w in warp_info if w != (self.ly, self.lx)]
                if opts: dest = self.generator.rng.choice(opts)

            if dest:
                # Ensure destination isn't blocked by a random stone at this exact moment
                stone_at_dest = any(s[0] == dest[0] and s[1] == dest[1] for s in self.curr_room.stones)
                if not stone_at_dest:
                    self.ly, self.lx = dest

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create base grid
        full_grid = np.full((self.generator.max_room_dim, self.generator.max_room_dim), TILE_WALL, dtype=np.int8)
        h, w = self.curr_room.h, self.curr_room.w
        copy_h, copy_w = min(h, self.generator.max_room_dim), min(w, self.generator.max_room_dim)

        room_grid = self.curr_room.grid_array[:copy_h, :copy_w].copy()

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