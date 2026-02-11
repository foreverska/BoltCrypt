# Environment Guide

## Overview

BoltCrypt is a procedurally generated dungeon exploration environment designed for Reinforcement Learning research. It combines navigation, puzzle-solving, and exploration challenges in a compact, efficient package.

## Core Concepts

### Room-Based Dungeon Structure

Dungeons are graphs of interconnected rooms:
- Each room is a grid of tiles (4-10 tiles wide/tall)
- Rooms connect via **doors** placed on walls
- The dungeon uses a global coordinate system where each room has a unique `(x, y)` position

### Agent Navigation

The agent moves within and between rooms:
- **Local Position:** `(lx, ly)` within the current room
- **Global Position:** `(gx, gy)` of the current room in the dungeon graph
- **Actions:** 4 cardinal directions (North, South, East, West)

### Observation Space

The environment provides a dictionary observation:

```python
{
    'grid': np.ndarray,      # (max_room_dim × max_room_dim) tile grid
    'agent_pos': [lx, ly],   # Local position in current room
    'global_pos': [gx, gy],  # Current room coordinates
    'inventory': 0 or 1      # Key possession flag
}
```

**Grid Details:**
- Only shows the current room (padded with walls if smaller than max size)
- Static tiles: walls, doors, switches, exit, key
- Dynamic overlays: boulders, sailing stones (updated each step)
- Agent position is NOT in the grid (use `agent_pos`)

## Dungeon Generation

### Generation Pipeline

1. **Skeleton Path:** Build minimum-distance path from start to end
2. **Growth:** Add rooms until target count reached
3. **Exit Selection:** Choose a vault (1-door room) at valid distance
4. **Connectivity:** Add extra doors to create loops
5. **Puzzles:** Place puzzles in eligible rooms
6. **Key-Lock:** Optionally lock exit and hide key

### Configuration Parameters

```python
config = {
    'min_dist': 10,           # Minimum Manhattan distance to exit
    'max_dist': 15,           # Maximum distance for exit selection
    'mean_rooms': 30,         # Average room count
    'std_rooms': 5,           # Standard deviation
    'connectivity': 0.3,      # Loop probability (0=tree, 1=highly connected)
    'puzzle_density': 0.4,    # Puzzle placement probability
    'allowed_puzzles': [      # Puzzle types to use
        'boulder',
        'mapped_plates',
        'stone',
        'warp_cycle',
        'warp_rnd'
    ],
    'key_puzzle_prob': 0.5,   # Probability of key-lock mechanism
    'randomize_end_distance': True  # Randomize end from min to 2*min
}

env = BoltCrypt(generator_config=config)
```

### Distance Mechanics

**Without `max_dist` or `randomize_end_distance`:**
- Exit must be ≥ `min_dist` from start

**With `max_dist`:**
- Exit chosen from range `[min_dist, max_dist]`

**With `randomize_end_distance=True`:**
- Actual required distance randomized between `min_dist` and `2*min_dist`

## Episode Structure

### Initialization

```python
obs, info = env.reset(seed=42)  # Optional seed for reproducibility
```

- Generates new dungeon
- Places agent in start room (center position)
- Resets inventory to empty

### Step Cycle

```python
obs, reward, done, truncated, info = env.step(action)
```

**Order of Operations:**
1. **Pre-Move:** Sailing stones move randomly
2. **Action Parsing:** Convert action to direction
3. **Movement Attempt:** Check collisions, doors, locks
4. **Interactions:** Pick up key, push boulders, teleport
5. **Puzzle Check:** Update door lock states
6. **Reward Calculation:** Based on events
7. **Observation:** Generate new grid with updated positions

### Termination

Episode ends when:
- Agent reaches the EXIT tile (`done=True`, `reward=+10.0`)
- Agent calls EXIT command (NaturalLanguage wrapper only)

Episodes do NOT have a max step limit by default (no truncation).

## Movement Mechanics

### Within-Room Movement

Agent moves within room boundaries:
- **Walls:** Block movement
- **Floors:** Free movement
- **Doors:** Transition to adjacent room (if unlocked)
- **Boulders:** Pushable if space behind is valid
- **Sailing Stones:** Act as walls
- **Keys:** Picked up, removed from grid
- **Exit:** Episode ends (+10 reward)

### Room Transitions

When moving through a door:
1. Check puzzle locks (current room)
2. Check key locks (target room)
3. If unlocked, update `global_pos` and enter opposite door of target room
4. Agent appears at door position in new room

**Door Lock Conditions:**
- **Boulder Puzzle:** All switches must have boulders
- **Boulder Plates:** Boulder must be on correct plate for that door
- **Key Lock:** Inventory must contain key

### Boulder Pushing

When moving into a boulder:
1. Calculate push destination (same direction)
2. Check if destination is valid (not wall/door/exit/boulder/stone)
3. If valid, boulder moves, agent follows
4. Puzzle state rechecked

**Boulder Rules:**
- Can't push boulders out of bounds (need 1-tile safety margin from walls)
- Can't push into other boulders
- Can't push through doors

### Teleportation

**Warp Cycle Rooms:**
- Stepping on any floor tile teleports to next in cycle
- Forms a Hamilton cycle of all floor tiles
- Deterministic but requires learning the permutation

**Warp Random Rooms:**
- Stepping on any floor tile teleports randomly
- Non-deterministic
- Breaks next-state prediction

## Reward Structure

### Default Rewards

```python
BoltCrypt(puzzle_bonus=True, key_bonus=True)
```

| Event | Reward |
|-------|--------|
| Time step | -0.01 |
| Collect key | +1.0 (if key_bonus) |
| Solve boulder puzzle | +1.0 (if puzzle_bonus, first time only) |
| Reach exit | +10.0 |

### Customizing Rewards

**Disable Bonus Rewards:**
```python
env = BoltCrypt(puzzle_bonus=False, key_bonus=False)
# Only +10 for exit, -0.01 per step
```

**Add Discovery Rewards:**
```python
env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)
# +0.5 for each new room discovered
```

## Difficulty Tuning

### Easy Configuration

```python
easy_config = {
    'min_dist': 3,
    'mean_rooms': 5,
    'connectivity': 0.8,      # Lots of loops
    'puzzle_density': 0.0,    # No puzzles
    'key_puzzle_prob': 0.0    # No key required
}
```

### Medium Configuration

```python
medium_config = {
    'min_dist': 8,
    'mean_rooms': 15,
    'connectivity': 0.5,
    'puzzle_density': 0.3,
    'allowed_puzzles': ['boulder'],  # Only simple puzzles
    'key_puzzle_prob': 0.3
}
```

### Hard Configuration

```python
hard_config = {
    'min_dist': 15,
    'mean_rooms': 40,
    'connectivity': 0.1,      # Mostly tree structure
    'puzzle_density': 0.6,
    'allowed_puzzles': ['boulder', 'mapped_plates', 'stone', 'warp_cycle'],
    'key_puzzle_prob': 0.8,
    'randomize_end_distance': True
}
```

### Extreme (RND/ICM Killer)

```python
extreme_config = {
    'min_dist': 20,
    'mean_rooms': 50,
    'connectivity': 0.0,      # Pure tree
    'puzzle_density': 0.9,
    'allowed_puzzles': ['warp_rnd', 'stone'],  # Maximum chaos
    'key_puzzle_prob': 1.0
}
```

## Performance

BoltCrypt is designed for high throughput:
- Pure NumPy operations (no rendering overhead)
- Efficient room caching
- Minimal memory footprint

**Typical Performance:**
- ~10,000+ resets/second
- ~50,000+ steps/second

Run the benchmark:
```bash
python -m boltcrypt.examples.speed_test
```

## Common Patterns

### Basic Training Loop

```python
from boltcrypt.envs import BoltCrypt

env = BoltCrypt()
obs, info = env.reset()

for episode in range(1000):
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, trunc, info = env.step(action)
        agent.update(obs, reward, done)

    obs, info = env.reset()
```

### Curriculum Learning

```python
# Start easy, gradually increase difficulty
configs = [
    {'min_dist': 3, 'mean_rooms': 5},
    {'min_dist': 5, 'mean_rooms': 10},
    {'min_dist': 8, 'mean_rooms': 15, 'puzzle_density': 0.2},
    {'min_dist': 12, 'mean_rooms': 25, 'puzzle_density': 0.4},
]

for config in configs:
    env = BoltCrypt(generator_config=config)
    train_agent(env, episodes=500)
```

### Wrapper Stacking

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import RoomDiscoveryReward, FogOfWar

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.2)
env = FogOfWar(env, vision_range=2)

# Agent now has:
# - 5x5 vision window
# - Exploration bonuses
```

## Debugging Tips

### Visualize Current State

```python
obs, info = env.reset()

# Current room
print(f"Global position: {obs['global_pos']}")
print(f"Agent position: {obs['agent_pos']}")
print(f"Has key: {bool(obs['inventory'])}")

# Room grid
print(obs['grid'])

# Check puzzle state
room = env.curr_room
print(f"Puzzle type: {room.puzzle_type}")
print(f"Is solved: {room.is_solved}")
print(f"Boulders: {room.boulders}")
print(f"Switches: {room.switches}")
```

### Test Generation

```python
# Generate multiple dungeons to verify configuration
for i in range(10):
    obs, info = env.reset(seed=i)
    gen = env.generator
    print(f"Seed {i}: {len(gen.grid)} rooms, exit at {gen.end_pos}")
```

### Play Manually

```bash
# Pygame interface
python -m boltcrypt.game.boltcrypt_game

# Terminal interface
python -m boltcrypt.game.boltcrypt_cli
```
