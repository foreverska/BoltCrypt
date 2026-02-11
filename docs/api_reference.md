# API Reference

## Core Environment

### BoltCrypt

**Location:** `boltcrypt.envs.BoltCrypt`

OpenAI Gymnasium environment for procedurally generated dungeon exploration.

#### Constructor

```python
BoltCrypt(generator_config=None, puzzle_bonus=True, key_bonus=True)
```

**Parameters:**
- `generator_config` (dict, optional): Configuration dictionary for DungeonGenerator
  - `min_dist` (int): Minimum Manhattan distance from start to exit (default: 5)
  - `max_dist` (int, optional): Maximum distance for exit selection (default: None = use min_dist)
  - `mean_rooms` (int): Average number of rooms (default: 15)
  - `std_rooms` (int): Standard deviation for room count (default: 2)
  - `connectivity` (float): Probability of adding loops, 0-1 (default: 0.5)
  - `puzzle_density` (float): Probability of puzzles in rooms, 0-1 (default: 0.0)
  - `allowed_puzzles` (list): List of puzzle type strings (default: all types)
  - `key_puzzle_prob` (float): Probability of key-lock puzzle, 0-1 (default: 0.0)
  - `randomize_end_distance` (bool): Randomize distance from min_dist to 2*min_dist
- `puzzle_bonus` (bool): Grant +1.0 reward for solving boulder puzzles (default: True)
- `key_bonus` (bool): Grant +1.0 reward for collecting keys (default: True)

#### Action Space

`Discrete(4)` - Four directional actions:
- `0`: North (up)
- `1`: South (down)
- `2`: East (right)
- `3`: West (left)

#### Observation Space

`Dict` with the following keys:
- `grid`: `Box(0, 9, shape=(max_room_dim, max_room_dim), dtype=int8)` - Room tile grid
- `agent_pos`: `Box(0, max_room_dim, shape=(2,), dtype=int32)` - Local (x, y) position
- `global_pos`: `Box(-1000, 1000, shape=(2,), dtype=int32)` - Global room coordinates
- `inventory`: `Discrete(2)` - Binary flag (1 if holding key, 0 otherwise)

#### Tile Types

| Value | Constant | Description |
|-------|----------|-------------|
| 0 | TILE_EMPTY | Walkable floor |
| 1 | TILE_WALL | Impassable wall |
| 2 | TILE_DOOR | Room transition |
| 3 | TILE_EXIT | Goal tile |
| 4 | TILE_SWITCH | Pressure plate |
| 5 | TILE_BOULDER | Pushable boulder |
| 6 | TILE_KEY | Collectible key |
| 7 | TILE_STONE | Sailing stone (RND obstacle) |
| 8 | TILE_WARP | Warp tile (ICM teleporter) |

#### Rewards

- `-0.01` per step (encourages efficiency)
- `+1.0` for collecting key (if `key_bonus=True`)
- `+1.0` for solving boulder puzzle (if `puzzle_bonus=True`)
- `+10.0` for reaching exit (terminates episode)

#### Methods

##### reset()

```python
reset(seed=None, options=None) -> tuple[dict, dict]
```

Reset environment and generate a new dungeon.

**Returns:** `(observation, info)`

##### step()

```python
step(action: int) -> tuple[dict, float, bool, bool, dict]
```

Execute one environment step.

**Returns:** `(observation, reward, done, truncated, info)`

---

## Wrappers

### NaturalLanguage

**Location:** `boltcrypt.wrapper.NaturalLanguage`

Converts observations to natural language descriptions for LLM agents.

#### Constructor

```python
NaturalLanguage(env)
```

**Parameters:**
- `env`: BoltCrypt environment instance

#### Text Commands

Instead of numeric actions, accepts string commands:

**Directional:**
- `NORTH`, `N`, `UP` → Move north
- `SOUTH`, `S`, `DOWN` → Move south
- `EAST`, `E`, `RIGHT` → Move east
- `WEST`, `W`, `LEFT` → Move west

**Utility:**
- `LOOK` → Re-describe current room
- `RESET` → Generate new dungeon
- `EXIT`, `QUIT` → Terminate episode

#### Observation Format

Returns multi-section text descriptions:
1. **Event Log** - What just happened
2. **Narrative** - Atmospheric flavor text (on room entry)
3. **Situation Report** - Coordinates, doors, objects with relative positions
4. **Visual Scan** - 3x3 ASCII grid around player

#### Example Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage

env = BoltCrypt()
env = NaturalLanguage(env)

obs, info = env.reset()
print(obs)  # Text description

obs, reward, done, trunc, info = env.step("NORTH")
print(obs)  # Updated text description
```

---

### FogOfWar

**Location:** `boltcrypt.wrapper.FogOfWar`

Limits agent vision to a local window, creating partial observability.

#### Constructor

```python
FogOfWar(env, vision_range=1)
```

**Parameters:**
- `env`: BoltCrypt environment instance
- `vision_range` (int): Tiles visible in each direction (default: 1 = 3x3 window)

#### Observation Modification

Replaces the full room `grid` with a `(2*vision_range + 1) × (2*vision_range + 1)` window centered on the agent. Areas beyond vision range are filled with walls.

#### Example Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import FogOfWar

env = BoltCrypt()
env = FogOfWar(env, vision_range=2)  # 5x5 vision window

obs, info = env.reset()
print(obs['grid'].shape)  # (5, 5) instead of full room
```

---

### RoomDiscoveryReward

**Location:** `boltcrypt.wrapper.RoomDiscoveryReward`

Grants bonus rewards for entering new rooms, encouraging exploration.

#### Constructor

```python
RoomDiscoveryReward(env, discovery_reward=0.1)
```

**Parameters:**
- `env`: BoltCrypt environment instance
- `discovery_reward` (float): Bonus reward per new room discovered (default: 0.1)

#### Behavior

Tracks visited rooms by `global_pos`. First time entering a room adds `discovery_reward` to the step reward.

#### Example Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import RoomDiscoveryReward

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)

obs, info = env.reset()
obs, reward, done, trunc, info = env.step(0)
# reward includes +0.5 if entered new room
```

---

## Dungeon Generator

### DungeonGenerator

**Location:** `boltcrypt.env.boltcrypt.DungeonGenerator`

Procedural dungeon generator creating connected room graphs.

#### Constructor

```python
DungeonGenerator(
    seed=None,
    min_dist=5,
    max_dist=None,
    mean_rooms=15,
    std_rooms=2,
    connectivity=0.5,
    puzzle_density=0.0,
    allowed_puzzles=None,
    key_puzzle_prob=0.0,
    min_room_dim=4,
    max_room_dim=10
)
```

See BoltCrypt constructor for parameter descriptions.

#### Methods

##### seed()

```python
seed(seed=None)
```

Set random seed for reproducible generation.

##### generate()

```python
generate()
```

Generate complete dungeon layout. Automatically called by BoltCrypt.reset().

---

## Room

**Location:** `boltcrypt.envs.boltcrypt.Room`

Represents a single dungeon room with puzzle mechanics.

#### Attributes

- `x`, `y` (int): Global dungeon coordinates
- `w`, `h` (int): Room dimensions
- `is_start` (bool): Starting room flag
- `is_exit` (bool): Exit room flag
- `is_locked` (bool): Requires key to enter
- `has_key` (bool): Contains a key
- `doors` (dict): Direction → offset mapping
- `puzzle_type` (PuzzleType): Type of puzzle in room
- `is_solved` (bool): Puzzle solution status
- `boulders` (list): Dynamic boulder positions [[y, x], ...]
- `stones` (list): Sailing stone positions [[y, x], ...]
- `switches` (set): Pressure plate positions {(y, x), ...}
- `warps` (dict): Warp tile mappings

#### Properties

##### grid_array

```python
@property
grid_array -> np.ndarray
```

Cached static grid background (walls, doors, switches, etc.). Dynamic entities like boulders are overlaid at render time.

---

## Puzzle Types

### PuzzleType Enum

**Location:** `boltcrypt.envs.boltcrypt.PuzzleType`

```python
class PuzzleType(Enum):
    NONE = "none"
    BOULDER = "boulder"
    BOULDER_PLATES = "mapped_plates"
    STONE = "stone"
    WARP_CYCLE = "warp_cycle"
    WARP_RND = "warp_rnd"
```

**Descriptions:**
- `NONE`: No puzzle
- `BOULDER`: Push boulders onto all switches to unlock doors
- `BOULDER_PLATES`: One boulder, multiple plates - each plate unlocks a different door
- `STONE`: Randomly moving obstacles (RND challenge)
- `WARP_CYCLE`: Hamilton cycle teleporters (ICM challenge)
- `WARP_RND`: Random teleportation chaos (ICM++ challenge)
