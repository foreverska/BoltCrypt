import numpy as np

from boltcrypt.envs.defines import *

class Room:
    """
    Represents a single room in the dungeon with puzzle mechanics.

    Rooms can contain various puzzle types (boulders, pressure plates, teleporters, moving obstacles)
    and may have doors connecting to adjacent rooms.
    """
    __slots__ = ['x', 'y', 'w', 'h', 'is_start', 'is_exit', 'is_locked', 'has_key',
                 'doors', '_cached_grid', 'switches', 'columns', 'boulders', 'stones', 'warps',
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
        self.columns = set()
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

    def __gen_boulder_puzzle(self, rng, candidates):
        """
        Generate a boulder puzzle configuration for the room.

        :param rng: Random number generator for puzzle generation
        :param candidates: List of candidate positions for switches and boulders
        :return: None
        """
        max_elements = min(self.w - 4, self.h - 4)
        num_elements = rng.randint(1, max(1, min(3, max_elements)))
        chosen = rng.sample(candidates, num_elements * 2)
        for i in range(num_elements):
            self.switches.add(chosen[i])
            self.boulders.append(list(chosen[i + num_elements]))
        self.is_solved = False


    def __gen_mapped_plates_puzzle(self, rng, candidates):
        """
        Generate a boulder puzzle with mapped pressure plates configuration for the room.

        :param rng: Random number generator for puzzle generation
        :param candidates: List of candidate positions for switches and boulders
        :return: None
        """
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

    def __gen_stone_puzzle(self, rng, candidates):
        """
        Generate a sailing stones puzzle configuration for the room.

        :param rng: Random number generator for puzzle generation
        :param candidates: List of candidate positions for stones
        :return: None
        """
        # RND Sailing Stones
        num_stones = rng.randint(1, 3)
        # Ensure we don't block everything, pick sparse locations
        chosen = rng.sample(candidates, min(num_stones, len(candidates)))
        self.stones = [list(pos) for pos in chosen]
        # Stones don't need "solving", they are just hazards/noise
        self.is_solved = True

    def __gen_warp_puzzle(self, rng):
        """
        Generate a warp puzzle configuration for the room.

        :param rng: Random number generator for puzzle generation
        :return: None
        """
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
        if self.is_start or self.is_exit or self.has_key:
            return
        if rng.random() > puzzle_density:
            return
        if not allowed_types:
            return

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

        if self.puzzle_type == PuzzleType.BOULDER:
            self.__gen_boulder_puzzle(rng, candidates)
        elif self.puzzle_type == PuzzleType.BOULDER_PLATES:
            self.__gen_mapped_plates_puzzle(rng, candidates)
        elif self.puzzle_type == PuzzleType.STONE:
            self.__gen_stone_puzzle(rng, candidates)
        elif self.puzzle_type in [PuzzleType.WARP_CYCLE, PuzzleType.WARP_RND]:
            self.__gen_warp_puzzle(rng)

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
                if self.grid_array[ny, nx] in [TILE_WALL, TILE_COLUMN, TILE_DOOR, TILE_EXIT]: continue

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

        for (sy, sx) in self.columns:
            grid[sy, sx] = TILE_COLUMN

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
        Check if the room's puzzle is solved and update the puzzle state.

        Returns:
            bool: True if the puzzle is solved or has no puzzle, False otherwise
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