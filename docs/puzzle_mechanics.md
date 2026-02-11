# Puzzle Mechanics Guide

BoltCrypt features five distinct puzzle types, each presenting unique challenges for RL agents.

---

## Overview

Puzzles are optional room obstacles that lock doors until solved (or navigated correctly). They test different agent capabilities:

- **BOULDER:** Planning and goal-directed behavior
- **BOULDER_PLATES:** State-dependent navigation (mapped plates)
- **STONE:** Stochasticity and obstacle avoidance (RND challenge)
- **WARP_CYCLE:** Deterministic permutation learning (ICM challenge)
- **WARP_RND:** Random teleportation chaos (ICM++ challenge)

---

## BOULDER (Classic Sokoban)

### Description

Push boulders onto pressure plates to unlock all doors in the room.

### Mechanics

- **Switches (🔴):** Fixed pressure plates on floor
- **Boulders (🟤):** Pushable blocks
- **Goal:** Cover ALL switches with boulders
- **Lock State:** Doors locked until puzzle solved

### Rules

1. Agent can push boulders by walking into them
2. Boulder moves one tile in push direction
3. Boulders cannot be pushed:
   - Into walls, doors, or exit
   - Into other boulders
   - Into sailing stones
   - Out of safety margin (1 tile from wall)
4. Once all switches covered, doors unlock
5. Puzzle remains solved even if boulders move off switches

### Example Layout

```
#########
#  @  _ #
#       #
# _   @ #
#       #
#########
```

`@` = Boulder, `_` = Switch

**Solution:** Push each boulder onto a switch.

### Agent Challenges

- **Planning:** May require multi-step push sequences
- **Irreversibility:** Wrong pushes can make puzzle unsolvable
- **Spatial Reasoning:** Understanding push mechanics
- **Deadlock Detection:** Recognizing unsolvable states

### Generation

- 1-3 boulder/switch pairs per room
- Pairs placed randomly in valid interior positions
- No guarantee puzzle is solvable (adds challenge)

### Rewards

If `puzzle_bonus=True`:
- **First solve:** +1.0 reward
- **Subsequent solves:** No bonus (tracked via `previously_solved`)

### Testing Boulder Puzzles

```python
config = {
    'puzzle_density': 1.0,  # Every room
    'allowed_puzzles': ['boulder']
}
env = BoltCrypt(generator_config=config, puzzle_bonus=True)

obs, _ = env.reset()
room = env.curr_room

print(f"Boulders: {room.boulders}")
print(f"Switches: {room.switches}")
print(f"Solved: {room.is_solved}")
```

---

## BOULDER_PLATES (Mapped Pressure Plates)

### Description

One boulder, multiple pressure plates. Each plate unlocks a specific door.

### Mechanics

- **One Boulder:** Shared between all plates
- **Multiple Plates:** One per door in the room
- **Mapping:** Each plate unlocks exactly one door
- **State-Dependent:** Which door is open depends on boulder position

### Rules

1. Boulder must be on a plate to unlock ANY door
2. Only the door mapped to the active plate is unlocked
3. To change accessible door, push boulder to different plate
4. If boulder is NOT on a plate, ALL doors locked
5. Puzzle is never "solved" (always in flux)

### Example

Room with 3 doors (North, East, West):

```
Plate A (position 2,3) → Unlocks NORTH door
Plate B (position 4,5) → Unlocks EAST door
Plate C (position 3,2) → Unlocks WEST door
```

**Scenario 1:** Boulder on Plate A
- Can exit NORTH
- EAST and WEST locked

**Scenario 2:** Boulder on Plate B
- Can exit EAST
- NORTH and WEST locked

**Scenario 3:** Boulder NOT on any plate
- ALL doors locked

### Agent Challenges

- **Planning:** Must push boulder to correct plate for desired exit
- **Backtracking:** May need to revisit room from different entrance
- **State Tracking:** Remembering which plate unlocks which door
- **Non-Markovian:** Optimal action depends on intended destination

### Generation

- Number of plates = number of doors
- Random mapping between plates and doors
- Boulder starts at random position (may or may not be on a plate)

### Use Case

Tests **context-dependent decision making** - same observation requires different actions depending on goal.

### Testing Mapped Plates

```python
config = {
    'puzzle_density': 1.0,
    'allowed_puzzles': ['mapped_plates']
}
env = BoltCrypt(generator_config=config)

obs, _ = env.reset()
room = env.curr_room

print(f"Plate → Door mapping: {room.plate_door_map}")
print(f"Active plate: {room.active_plate}")
print(f"Doors: {list(room.doors.keys())}")
```

---

## STONE (Sailing Stones - RND Killer)

### Description

Randomly moving obstacles that create stochastic, unpredictable navigation.

### Mechanics

- **Sailing Stones (⚫):** Autonomous moving obstacles
- **Random Movement:** Each step, stones move in random direction
- **Blocking:** Stones act like walls (can't push them)
- **Collision Avoidance:** Stones don't move into walls, doors, boulders, or agent

### Rules

1. Every step, BEFORE agent moves, each stone attempts random movement
2. Stone picks random direction (N/S/E/W), tries to move
3. If destination is blocked (wall/door/other stone/boulder/agent), stone stays
4. Agent cannot push stones
5. Stones provide visual noise in observation

### Challenges

**Random Network Distillation (RND) Killer:**
- Creates **aleatoric uncertainty** (irreducible randomness)
- Breaks curiosity-driven exploration methods that rely on deterministic state transitions
- Agents must learn to ignore stone positions (or treat them as noise)

### Agent Difficulties

- **Non-Deterministic:** Same state + action → different next states
- **Exploration Confusion:** RND/ICM methods see constant "novelty"
- **Prediction Failure:** Forward models can't predict stone positions
- **Visual Noise:** Stones clutter observations without strategic value

### Generation

- 1-3 stones per room
- Placed randomly on valid floor tiles
- No puzzle to "solve" (stones are hazards)

### Strategic Implications

Stones don't lock doors - they just make navigation unpredictable:
- Might block intended path
- Might randomly clear path
- Agent must adapt on-the-fly

### Testing Stone Puzzles

```python
config = {
    'puzzle_density': 0.8,
    'allowed_puzzles': ['stone']
}
env = BoltCrypt(generator_config=config)

obs, _ = env.reset()
room = env.curr_room

print(f"Stone positions: {room.stones}")

# Stones move each step
for _ in range(5):
    obs, _, _, _, _ = env.step(0)
    print(f"Stones now: {env.curr_room.stones}")
```

---

## WARP_CYCLE (Hamilton Cycle - ICM Killer)

### Description

Every floor tile teleports to next tile in a deterministic cycle covering all positions.

### Mechanics

- **Hamilton Cycle:** Closed loop through ALL floor tiles
- **Warp Tiles (🌀):** Every floor tile is a warp
- **Deterministic:** Same tile → same destination (always)
- **Full Coverage:** Cycle visits every tile exactly once before looping

### Rules

1. Stepping on tile `(y, x)` instantly teleports to next tile in cycle
2. Cycle is randomized per room generation (different each reset)
3. Cycle forms complete permutation of room topology
4. To navigate, agent must LEARN the specific permutation

### Challenges

**Intrinsic Curiosity Module (ICM) Killer:**
- Creates **epistemic uncertainty** (reducible via learning)
- BUT permutation is room-specific, doesn't transfer
- ICM sees constant "surprise" as agent teleports unexpectedly
- Agent must memorize exact tile mappings (high memory load)

### Example

Small 3x3 room (9 floor tiles):

```
Cycle: A→B→C→D→E→F→G→H→I→A

Grid positions:
A(1,1) B(1,2) C(1,3)
D(2,1) E(2,2) F(2,3)
G(3,1) H(3,2) I(3,3)
```

Stepping on any tile warps you to the next:
- Step on A → Warp to B
- Step on B → Warp to C
- ...
- Step on I → Warp to A

**To reach destination:** Must learn the permutation and plan warp sequence.

### Agent Difficulties

- **Permutation Learning:** Must discover cycle structure
- **Memory:** Cycle is room-specific (no transfer)
- **Planning:** Multi-hop teleportation paths
- **Spatial Confusion:** Topology is non-Euclidean

### Strategic Depth

Unlike STONE (random), WARP_CYCLE is **deterministic but complex:**
- Agent CAN learn optimal policy
- Requires extensive exploration/memorization
- Each room has unique permutation

### Generation

```python
# Pseudocode
candidates = all_floor_tiles
shuffle(candidates)  # CRITICAL for chaos

for i in range(len(candidates)):
    src = candidates[i]
    dst = candidates[(i+1) % len(candidates)]
    warps[src] = dst
```

Shuffling ensures cycle isn't just linear path.

### Testing Warp Cycle

```python
config = {
    'puzzle_density': 1.0,
    'allowed_puzzles': ['warp_cycle']
}
env = BoltCrypt(generator_config=config)

obs, _ = env.reset()
room = env.curr_room

print(f"Warp mappings: {room.warps}")
print(f"Cycle length: {len(room.warps)}")

# Verify it's a cycle
visited = set()
current = list(room.warps.keys())[0]
for _ in range(len(room.warps) + 1):
    visited.add(current)
    current = room.warps[current]
    print(f"→ {current}")
```

---

## WARP_RND (Random Teleportation - ICM++ Killer)

### Description

Every floor tile teleports to a RANDOM floor tile on each step.

### Mechanics

- **Total Chaos:** No deterministic pattern
- **Random Destination:** Each warp picks random tile from all floor tiles
- **Non-Deterministic:** Same tile → different destinations
- **Unpredictable:** Impossible to learn or predict

### Rules

1. Stepping on any floor tile warps to random floor tile
2. Destination changes every time (not deterministic like WARP_CYCLE)
3. Can warp to same tile (no progress)
4. No memory or learning helps

### Challenges

**ICM++ Killer (Maximum Chaos):**
- Combines aleatoric AND epistemic uncertainty
- Completely breaks prediction-based exploration
- Forward models see infinite entropy
- Agents must rely on luck or exhaustive search

### Agent Difficulties

- **Zero Predictability:** No pattern to learn
- **Exploration Failure:** RND/ICM completely broken
- **Navigation Impossibility:** Can't plan paths
- **Blind Search:** Only option is random walk

### Strategic Implications

Rooms with WARP_RND are nearly unsolvable:
- Agent must get extremely lucky
- May take thousands of steps
- Tests agent's ability to handle irreducible randomness

### Generation

```python
# Pseudocode
candidates = all_floor_tiles

for tile in candidates:
    warps[tile] = candidates  # List of ALL tiles

# On step:
dest = random.choice(warps[current_tile])
```

Every warp rolls the dice.

### Testing Warp Random

```python
config = {
    'puzzle_density': 0.3,  # Don't overdo it
    'allowed_puzzles': ['warp_rnd']
}
env = BoltCrypt(generator_config=config)

obs, _ = env.reset()
room = env.curr_room

print(f"Warp mode: {room.warp_mode}")
print(f"Each tile maps to: {len(room.warps[list(room.warps.keys())[0]])} options")

# Test randomness
start_pos = (env.ly, env.lx)
destinations = []
for _ in range(10):
    obs, _ = env.reset()
    env.ly, env.lx = start_pos
    # Trigger warp
    destinations.append(env.curr_room.warps[start_pos])

print(f"Same tile warped to: {len(set(map(tuple, destinations)))} different places")
```

---

## Puzzle Combinations

### Configuration

```python
# Enable specific puzzles
config = {
    'puzzle_density': 0.5,
    'allowed_puzzles': ['boulder', 'stone']
}

# Enable all puzzles
config = {
    'puzzle_density': 0.7,
    'allowed_puzzles': ['boulder', 'mapped_plates', 'stone', 'warp_cycle', 'warp_rnd']
}

# Only navigation (no puzzles)
config = {
    'puzzle_density': 0.0
}
```

### Difficulty Progression

**Level 1: Pure Navigation**
```python
allowed_puzzles = []
```

**Level 2: Simple Puzzles**
```python
allowed_puzzles = ['boulder']
```

**Level 3: Complex Puzzles**
```python
allowed_puzzles = ['boulder', 'mapped_plates']
```

**Level 4: Stochastic Challenges**
```python
allowed_puzzles = ['boulder', 'stone']
```

**Level 5: Exploration Killers**
```python
allowed_puzzles = ['warp_cycle']
```

**Level 6: Chaos Mode**
```python
allowed_puzzles = ['stone', 'warp_rnd']
```

---

## Research Applications

### Testing Curiosity Methods

**Hypothesis:** RND/ICM fail on stochastic environments.

```python
# Baseline (deterministic)
env_det = BoltCrypt({'allowed_puzzles': ['boulder']})

# Stochastic challenge
env_stoch = BoltCrypt({'allowed_puzzles': ['stone', 'warp_rnd']})

# Compare RND performance
rnd_agent_det = train_rnd(env_det)
rnd_agent_stoch = train_rnd(env_stoch)
```

### Testing Memory

**Hypothesis:** WARP_CYCLE requires memory.

```python
# Memoryless agent (MLP)
mlp_agent = MLP_Policy()

# Memory agent (LSTM)
lstm_agent = LSTM_Policy()

env = BoltCrypt({'allowed_puzzles': ['warp_cycle']})

# LSTM should vastly outperform MLP
```

### Testing Generalization

**Hypothesis:** Puzzle-solving transfers across rooms.

```python
# Train on boulders
train_env = BoltCrypt({'allowed_puzzles': ['boulder']})
agent = train(train_env)

# Test on different puzzle
test_env = BoltCrypt({'allowed_puzzles': ['mapped_plates']})
evaluate(agent, test_env)
```

---

## Debugging Puzzles

### Check Puzzle Type

```python
obs, _ = env.reset()
room = env.curr_room

print(f"Puzzle: {room.puzzle_type}")
print(f"Solved: {room.is_solved}")
```

### Visualize Puzzle State

```python
import matplotlib.pyplot as plt

obs, _ = env.reset()
grid = obs['grid']

plt.imshow(grid, cmap='tab10')
plt.colorbar()
plt.title(f"Puzzle: {env.curr_room.puzzle_type.name}")
plt.show()
```

### Force Puzzle Solve

```python
# For testing downstream behavior
obs, _ = env.reset()
room = env.curr_room

if room.puzzle_type.name == 'BOULDER':
    # Place boulders on switches
    for i, switch in enumerate(room.switches):
        if i < len(room.boulders):
            room.boulders[i] = list(switch)
    room.check_solved()
    print(f"Manually solved: {room.is_solved}")
```

---

## Summary

| Puzzle | Type | Locks Doors | Challenge | Kills |
|--------|------|-------------|-----------|-------|
| BOULDER | Planning | Yes | Goal-directed behavior | - |
| BOULDER_PLATES | State-Dependent | Partial | Context switching | - |
| STONE | Stochastic | No | Aleatoric uncertainty | RND |
| WARP_CYCLE | Permutation | No | Epistemic uncertainty | ICM |
| WARP_RND | Chaos | No | Total randomness | ICM++ |

Choose puzzle types based on research goals!
