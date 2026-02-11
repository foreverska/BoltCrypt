# Wrappers Guide

BoltCrypt includes three powerful wrappers to modify the environment for different research paradigms.

## Overview

Wrappers follow the OpenAI Gymnasium pattern:

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage, FogOfWar, RoomDiscoveryReward

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)
env = FogOfWar(env, vision_range=2)
# Wrappers stack - inner applied first
```

---

## NaturalLanguage

**Purpose:** Convert grid observations to rich text descriptions for LLM agents.

### Features

- **Text Observations:** Replaces numeric grids with narrative descriptions
- **Text Actions:** Accept commands like "NORTH" instead of action integers
- **Event Logging:** Tracks what happened each step
- **Flavor Text:** Atmospheric descriptions on room entry
- **Tactical Info:** Object positions relative to player
- **3x3 ASCII View:** Simple visual representation

### Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage

env = BoltCrypt()
env = NaturalLanguage(env)

obs, info = env.reset()
print(obs)
```

**Example Output:**

```
> *** SYSTEM: PROTOCOL INITIATED ***
MISSION: Escape the Labyrinth.
INSTRUCTIONS: The dungeon is a grid. Solve puzzles to unlock doors. Find the Key. Reach the Exit.
Commands: NORTH, SOUTH, EAST, WEST, LOOK, RESET, EXIT.

--- NEW CHAMBER DISCOVERED ---
"A quiet, dusty chamber. The air is still."

--- SITUATION REPORT ---
LOCATION: Coordinate (5, 5) in a 8x7 room.
EXITS: NORTH Door at NORTH-Wall: (OPEN), EAST Door at EAST-Wall: (OPEN)
NEARBY: Heavy Boulder: 2 North, 1 East; Pressure Plate (EMPTY): 3 North

--- VISUAL SCAN (3x3) ---
[###]  .  [###]
[###][YOU]  .
[###]  .    .

Command >
```

### Commands

**Movement:**
- `NORTH`, `N`, `UP` - Move north
- `SOUTH`, `S`, `DOWN` - Move south
- `EAST`, `E`, `RIGHT` - Move east
- `WEST`, `W`, `LEFT` - Move west

**Utility:**
- `LOOK` - Re-describe the current room (full narrative)
- `RESET` - Generate new dungeon
- `EXIT`, `QUIT` - Terminate episode

### Text Structure

Each observation contains:

1. **Event Log**
   ```
   > You step through the door and enter a new room from the SOUTH.
   ```

2. **Narrative** (on room entry only)
   ```
   --- NEW CHAMBER DISCOVERED ---
   "The floor is scarred with deep grooves. Heavy boulders sit silently, waiting to be moved."
   ```

3. **Situation Report**
   ```
   --- SITUATION REPORT ---
   LOCATION: Coordinate (3, 4) in a 6x8 room.
   EXITS: NORTH Door at NORTH-Wall: (LOCKED), WEST Door at WEST-Wall: (OPEN)
   NEARBY: Pressure Plate (EMPTY): 2 North, 1 West; Heavy Boulder: 1 North; GOLDEN KEY: 3 East
   ```

4. **Visual Scan**
   ```
   --- VISUAL SCAN (3x3) ---
   [###][ @ ][###]
   [ _ ][YOU]  .
   [###]  .  [KEY]
   ```

### Visual Scan Legend

| Symbol | Meaning |
|--------|---------|
| `[YOU]` | Player position |
| `[###]` | Wall |
| `  .  ` | Floor |
| `[___]` | Door |
| `[OUT]` | Exit |
| `[ @ ]` | Boulder |
| `[OK!]` | Boulder on switch |
| `[ _ ]` | Empty pressure plate |
| `[KEY]` | Key |
| `[ * ]` | Sailing stone |

### Event Messages

The wrapper tracks and reports:

- **Room transitions:** "You step through the door and enter a new room from the WEST."
- **Movement:** "You move 1 step."
- **Blocking:** "BLOCKED. You bump into a solid obstacle."
- **Teleportation:** "WARP! The world twists—you reappear elsewhere!"
- **Boulder pushing:** "You heave the massive BOULDER forward."
- **Puzzle solution:** "CLICK! It locks into the pressure plate." / "MECHANISM: Gears grind and locks disengage."
- **Key pickup:** "SUCCESS! You acquired the GOLDEN KEY."
- **Victory:** "*** VICTORY! YOU HAVE ESCAPED! ***"

### Use Cases

**LLM Agents:**
```python
import anthropic

env = NaturalLanguage(BoltCrypt())
obs, _ = env.reset()

client = anthropic.Anthropic()
while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": obs}]
    )
    action = response.content[0].text.strip()
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
```

**Human Play:**
```python
env = NaturalLanguage(BoltCrypt())
obs, _ = env.reset()
print(obs)

while True:
    action = input().strip()
    obs, reward, done, _, _ = env.step(action)
    print(obs)
    if done:
        break
```

---

## FogOfWar

**Purpose:** Limit agent vision to create partial observability.

### Features

- Reduces grid observation to local window
- Forces agents to build internal memory/maps
- Configurable vision range
- Areas outside vision filled with walls

### Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import FogOfWar

env = BoltCrypt()
env = FogOfWar(env, vision_range=1)  # 3x3 window

obs, info = env.reset()
print(obs['grid'].shape)  # (3, 3) instead of (10, 10)
```

### Vision Range

The `vision_range` parameter controls visibility:

| vision_range | Window Size | Total Tiles Visible |
|--------------|-------------|---------------------|
| 1 (default) | 3×3 | 9 |
| 2 | 5×5 | 25 |
| 3 | 7×7 | 49 |
| 5 | 11×11 | 121 |

**Formula:** Window is `(2*vision_range + 1)²` tiles, centered on agent.

### Behavior

**What agent sees:**
- Tiles within vision range
- Exact tile types (walls, floors, boulders, etc.)
- Agent is always at center of returned grid

**What agent doesn't see:**
- Tiles beyond vision range (replaced with walls)
- Full room layout
- Door positions unless very close

**Other observations unchanged:**
- `agent_pos` still gives local room position
- `global_pos` still gives room coordinates
- `inventory` unchanged

### Challenges Introduced

1. **Memory Required:** Agent must remember visited areas
2. **Navigation Difficulty:** Can't see distant doors
3. **Puzzle Complexity:** May not see all switches/boulders at once
4. **Exploration:** Blind corners and dead ends

### Example Comparison

**Without FogOfWar:**
```python
env = BoltCrypt()
obs, _ = env.reset()
print(obs['grid'])
# Full 10x10 room visible
```

**With FogOfWar:**
```python
env = FogOfWar(BoltCrypt(), vision_range=1)
obs, _ = env.reset()
print(obs['grid'])
# Only 3x3 around agent
```

### Use Cases

**LSTM/Transformer Agents:**
```python
env = FogOfWar(BoltCrypt(), vision_range=2)
# Agent must maintain memory across steps
```

**Recurrent Policies:**
```python
# Partial observability forces temporal credit assignment
env = FogOfWar(BoltCrypt(), vision_range=1)
agent = RecurrentPPO(env)
```

**Memory Research:**
```python
# Test memory capacity with different vision ranges
for vr in [1, 2, 3]:
    env = FogOfWar(BoltCrypt(), vision_range=vr)
    test_agent(env)
```

---

## RoomDiscoveryReward

**Purpose:** Combat sparse rewards by incentivizing exploration.

### Features

- Bonus reward on first entry to each room
- Tracks visited rooms automatically
- Encourages graph exploration
- Configurable bonus amount

### Usage

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import RoomDiscoveryReward

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)

obs, info = env.reset()
obs, reward, done, _, _ = env.step(0)
# reward includes +0.5 if entered new room
```

### Reward Modification

**Base environment rewards:**
- `-0.01` per step
- `+1.0` for key/puzzle (if enabled)
- `+10.0` for exit

**With RoomDiscoveryReward:**
- All base rewards PLUS
- `+discovery_reward` first time entering each room

### Example

```python
config = {'min_dist': 10, 'mean_rooms': 20}
env = BoltCrypt(generator_config=config)
env = RoomDiscoveryReward(env, discovery_reward=0.2)

obs, _ = env.reset()

# Starting room already counted as visited
# Step into new room
obs, reward, _, _, _ = env.step(0)
# reward = -0.01 (step) + 0.2 (discovery) = 0.19

# Stay in same room
obs, reward, _, _, _ = env.step(1)
# reward = -0.01 (only step penalty)

# Enter another new room
obs, reward, _, _, _ = env.step(0)
# reward = -0.01 + 0.2 = 0.19
```

### Tuning Discovery Reward

**Low values (0.1 - 0.2):**
- Gentle exploration nudge
- Doesn't dominate other rewards
- Good for agents that already explore

**Medium values (0.5 - 1.0):**
- Strong exploration incentive
- Comparable to puzzle/key rewards
- Encourages thorough graph coverage

**High values (2.0+):**
- Exploration becomes primary objective
- May ignore puzzles/exit until all rooms found
- Useful for pure exploration research

### Use Cases

**Sparse Reward Environments:**
```python
# Large dungeon with few puzzles
config = {'min_dist': 20, 'mean_rooms': 50, 'puzzle_density': 0.1}
env = BoltCrypt(generator_config=config)
env = RoomDiscoveryReward(env, discovery_reward=1.0)
```

**Exploration Research:**
```python
# Measure exploration efficiency
env = RoomDiscoveryReward(BoltCrypt(), discovery_reward=0.5)

total_rooms = len(env.generator.grid)
visited_rooms = len(env.visited_rooms)
coverage = visited_rooms / total_rooms
```

**Curriculum Learning:**
```python
# Gradually reduce exploration bonus as agent improves
for phase in [1.0, 0.5, 0.2, 0.0]:
    env = RoomDiscoveryReward(BoltCrypt(), discovery_reward=phase)
    train_agent(env, episodes=1000)
```

---

## Combining Wrappers

Wrappers can be stacked for combined effects:

### Order Matters

```python
# FogOfWar → RoomDiscoveryReward
env = BoltCrypt()
env = FogOfWar(env, vision_range=1)
env = RoomDiscoveryReward(env, discovery_reward=0.5)
# Agent has limited vision AND exploration bonuses
```

### Common Combinations

**LLM with Exploration:**
```python
env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.3)
env = NaturalLanguage(env)
# Text descriptions with exploration incentives
```

**Partial Observability with Dense Rewards:**
```python
env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)
env = FogOfWar(env, vision_range=2)
# Limited vision but exploration helps
```

**Full Stack:**
```python
config = {'min_dist': 15, 'mean_rooms': 30, 'puzzle_density': 0.4}
env = BoltCrypt(generator_config=config)
env = RoomDiscoveryReward(env, discovery_reward=0.3)
env = FogOfWar(env, vision_range=2)
# Challenging: large dungeon, partial observability, exploration bonus
```

---

## Creating Custom Wrappers

You can create your own wrappers following the Gymnasium pattern:

### Observation Wrapper

```python
import gymnasium as gym

class CustomObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        # Modify observation dict
        obs['grid'] = transform(obs['grid'])
        return obs
```

### Reward Wrapper

```python
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        # Modify reward
        modified_reward = reward * 2

        return obs, modified_reward, done, trunc, info
```

### Action Wrapper

```python
class CustomActionWrapper(gym.ActionWrapper):
    def action(self, action):
        # Transform action before passing to envs
        return modified_action
```

### Example: Step Limit

```python
class StepLimit(gym.Wrapper):
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.steps += 1

        if self.steps >= self.max_steps:
            trunc = True

        return obs, reward, done, trunc, info
```

### Example: One-Hot Grid

```python
import numpy as np

class OneHotGrid(gym.ObservationWrapper):
    def __init__(self, env, num_tiles=9):
        super().__init__(env)
        self.num_tiles = num_tiles

    def observation(self, obs):
        grid = obs['grid']
        h, w = grid.shape

        # Convert to one-hot
        one_hot = np.zeros((h, w, self.num_tiles))
        for i in range(h):
            for j in range(w):
                tile = grid[i, j]
                if tile < self.num_tiles:
                    one_hot[i, j, tile] = 1

        obs['grid'] = one_hot
        return obs
```

Usage:
```python
env = BoltCrypt()
env = OneHotGrid(env)
# grid is now (H, W, 9) instead of (H, W)
```
