import enum

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
TILE_COLUMN = 9

# DIMENSIONS
MIN_ROOM_DIM = 4  # Bumped up slightly to fit puzzles
MAX_ROOM_DIM = 10


class Direction(enum.Enum):
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


class PuzzleType(enum.Enum):
    NONE = "none"
    BOULDER = "boulder"  # Push blocks to switches
    BOULDER_PLATES = "mapped_plates"  # One boulder, multiple pressure plates
    STONE = "stone"  # Randomly moving obstacles (RND Killer)
    WARP_CYCLE = "warp_cycle"  # Hamilton cycle teleporters (ICM Killer)
    WARP_RND = "warp_rnd"  # Degenerate random teleporters (ICM Killer++)