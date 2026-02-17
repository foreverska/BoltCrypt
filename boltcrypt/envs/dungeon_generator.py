import random
import numpy as np

from boltcrypt.envs.defines import *
from boltcrypt.envs.room import Room


class DungeonGenerator:
    """
    Procedural dungeon generator creating a connected graph of rooms with puzzles.

    Generates dungeons by:
    1. Building a skeleton path from start to a minimum distance
    2. Growing additional rooms to reach the target room count
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
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
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
        self.skeleton_end_pos = None  # Where a skeleton path ends

        if min_room_dim < MIN_ROOM_DIM:
            raise ValueError(f"min_room_dim must be >= {MIN_ROOM_DIM}")

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

        self._build_skeleton()
        self._grow_to_target(target_count)
        self._select_goal_room()
        self._apply_connectivity()
        self._apply_key_puzzle()

        # Find the room leading to the exit vault (if puzzle_required is enabled)
        vault_parent_room = None
        if self.config.get('puzzle_required') and self.end_pos in self.grid:
            exit_room = self.grid[self.end_pos]

            # Trace the door(s) of the exit room backward to find the connected parent
            for d in exit_room.doors:
                parent_pos = (self.end_pos[0] + d.value[0], self.end_pos[1] + d.value[1])
                if parent_pos in self.grid:
                    vault_parent_room = self.grid[parent_pos]
                    break  # Found the connected parent

            if vault_parent_room is not None:
                # Upgrade room size to ensure puzzle is visible
                vault_parent_room.w, vault_parent_room.h = max(vault_parent_room.w, 7), max(vault_parent_room.h, 7)
                vault_parent_room.freeze()
                vault_parent_room.setup_puzzle(self.rng, 1.0, self.allowed_puzzles)


        # Apply puzzles to all rooms
        for pos, r in self.grid.items():
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

        # Create rooms
        for i, pos in enumerate(path_coords):
            is_start = (pos == self.start_pos)
            w, h = self._get_random_dims()
            self.grid[pos] = Room(pos[0], pos[1], w, h, is_start, is_exit=False)
            if i > 0:
                prev = path_coords[i - 1]
                dx, dy = pos[0] - prev[0], pos[1] - prev[1]
                self._connect_rooms(self.grid[prev], self.grid[pos], Direction((dx, dy)))

        # Mark the last room as an exit for the sake of ensuring it remains a valid vault
        self.grid[path_coords[-1]].is_exit = True
        self.end_pos = path_coords[-1]

    def _grow_to_target(self, target_count):
        """
        Grow the dungeon efficiently using a dynamic frontier list.
        Start and end rooms are excluded from the initial frontier to keep them as vaults.
        """
        # 1. Initialize the frontier from the skeleton path
        candidates = []
        for pos in self.grid:
            # Enforce Vaults: Do not allow the start or end rooms to spawn new branches
            if pos == self.start_pos or pos == self.end_pos:
                continue

            for d in Direction:
                neighbor = (pos[0] + d.value[0], pos[1] + d.value[1])
                if neighbor not in self.grid:
                    candidates.append((neighbor, pos, d))

        # 2. Grow until we hit the target or run out of valid space
        while len(self.grid) < target_count and candidates:
            # O(1) random selection and removal from the frontier
            idx = self.rng.randrange(len(candidates))
            candidates[idx], candidates[-1] = candidates[-1], candidates[idx]
            new_pos, parent_pos, d = candidates.pop()

            # Multiple parents might have proposed this space. Skip if it was already built on.
            if new_pos in self.grid:
                continue

            # Create and place the new room
            w, h = self._get_random_dims()
            new_room = Room(new_pos[0], new_pos[1], w, h)
            self.grid[new_pos] = new_room
            self._connect_rooms(self.grid[parent_pos], new_room, d)

            # Add the new room's empty neighbors to the frontier
            for next_d in Direction:
                neighbor = (new_pos[0] + next_d.value[0], new_pos[1] + next_d.value[1])
                if neighbor not in self.grid:
                    candidates.append((neighbor, new_pos, next_d))

    def _select_goal_room(self):
        """
        Select the exit room from vault rooms (1-door rooms) within the distance range.

        Prioritizes dead-end rooms at valid distances (min_dist to max_dist) from start.
        Falls back to the skeleton end position if no valid vaults exist.
        """
        min_dist = self.config['min_dist']
        max_dist = self.config['max_dist']
        cur_end_pos = self.end_pos

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
            # mark the existing exit as no longer an exit
            self.grid[cur_end_pos].is_exit = False
            self.end_pos = self.rng.choice(vaults)
            self.grid[self.end_pos].is_exit = True



    def _apply_connectivity(self):
        """
        Add doors to adjacent rooms to create additional connectivity.
        Do not apply to start or exit rooms.
        :return: None
        """
        if self.config['connectivity'] <= 0: return
        for pos, room in self.grid.items():
            if room.is_exit or room.is_start: continue
            for d in Direction:
                n_pos = (pos[0] + d.value[0], pos[1] + d.value[1])
                if n_pos in self.grid and d not in room.doors:
                    neighbor = self.grid[n_pos]
                    if neighbor.is_exit or neighbor.is_start: continue
                    if self.rng.random() > self.config['connectivity']:
                        self._connect_rooms(room, neighbor, d)

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
            # Primary: Hide the key in a dead end to force exploration
            self.rng.choice(dead_ends).place_key(self.rng)
        else:
            # Fallback: No dead ends exist (rare, but possible in loops), pick a random inner room
            self.rng.choice(inner_rooms).place_key(self.rng)