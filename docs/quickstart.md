# Quick Start Guide

Get started with BoltCrypt in minutes!

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/BoltCrypt.git
cd BoltCrypt

# Install dependencies
pip install gymnasium pygame numpy matplotlib

# Install BoltCrypt
pip install -e .
```

### From PyPI (if published)

```bash
pip install boltcrypt
```

---

## Basic Usage

### Hello World

```python
from boltcrypt.envs import BoltCrypt

# Create environment
env = BoltCrypt()

# Reset to start episode
obs, info = env.reset()

# Take random actions
for _ in range(100):
    action = env.action_space.sample()  # Random action (0-3)
    obs, reward, done, truncated, info = env.step(action)

    if done:
        print("Reached exit!")
        break
```

### Understanding Observations

```python
obs, info = env.reset()

# Observation is a dictionary
print(obs.keys())  # ['grid', 'agent_pos', 'global_pos', 'inventory']

# Grid: current room layout
print(obs['grid'].shape)  # (10, 10) or smaller
print(obs['grid'])

# Agent position in room
print(obs['agent_pos'])  # [x, y]

# Room coordinates in dungeon
print(obs['global_pos'])  # [gx, gy]

# Key possession
print(obs['inventory'])  # 0 or 1
```

### Taking Actions

```python
# Actions are integers 0-3
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

obs, reward, done, truncated, info = env.step(NORTH)

# Rewards:
# -0.01 per step
# +1.0 for key/puzzle (optional)
# +10.0 for reaching exit
```

---

## Play Manually

### Pygame Interface

```bash
python -m boltcrypt.game.boltcrypt_game
```

**Controls:**
- Arrow keys: Move
- R: Reset dungeon
- ESC: Quit

### Terminal Interface

```bash
python -m boltcrypt.game.boltcrypt_cli
```

**Controls:**
- WASD or arrow keys: Move
- R: Reset dungeon
- Q: Quit

---

## Configuration Examples

### Simple Dungeon

```python
config = {
    'min_dist': 3,        # Close exit
    'mean_rooms': 5,      # Small dungeon
    'connectivity': 0.8,  # Lots of paths
    'puzzle_density': 0.0 # No puzzles
}

env = BoltCrypt(generator_config=config)
```

### Medium Challenge

```python
config = {
    'min_dist': 8,
    'mean_rooms': 15,
    'connectivity': 0.5,
    'puzzle_density': 0.3,
    'allowed_puzzles': ['boulder'],
    'key_puzzle_prob': 0.3
}

env = BoltCrypt(generator_config=config)
```

### Hard Dungeon

```python
config = {
    'min_dist': 15,
    'mean_rooms': 40,
    'connectivity': 0.1,  # Sparse connections
    'puzzle_density': 0.6,
    'allowed_puzzles': ['boulder', 'mapped_plates', 'stone'],
    'key_puzzle_prob': 0.8
}

env = BoltCrypt(generator_config=config)
```

---

## Using Wrappers

### Exploration Rewards

```python
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import RoomDiscoveryReward

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)

obs, info = env.reset()
obs, reward, done, truncated, info = env.step(0)
# reward includes +0.5 if new room discovered
```

### Limited Vision

```python
from boltcrypt.wrapper import FogOfWar

env = BoltCrypt()
env = FogOfWar(env, vision_range=2)  # 5x5 vision window

obs, info = env.reset()
print(obs['grid'].shape)  # (5, 5)
```

### Text Interface for LLMs

```python
from boltcrypt.wrapper import NaturalLanguage

env = BoltCrypt()
env = NaturalLanguage(env)

obs, info = env.reset()
print(obs)  # Text description

obs, reward, done, trunc, info = env.step("NORTH")
print(obs)  # Updated description
```

---

## Training a Simple Agent

### Random Agent

```python
from boltcrypt.envs import BoltCrypt

env = BoltCrypt()

for episode in range(10):
    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)

        total_reward += reward
        steps += 1

        if done:
            print(f"Episode {episode}: {total_reward:.2f} reward in {steps} steps")
            break
```

### Q-Learning Agent

See `boltcrypt/examples/tabular_q.py` for a complete tabular Q-learning implementation.

```bash
python -m boltcrypt.examples.tabular_q
```

### Using Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from boltcrypt.envs import BoltCrypt

# Flatten dict observation for SB3
from gymnasium.wrappers import FlattenObservation


def make_env():
    env = BoltCrypt()
    env = FlattenObservation(env)
    return env


env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones[0]:
        break
```

---

## Common Recipes

### Fixed Seed for Reproducibility

```python
env = BoltCrypt()

# Same dungeon every reset
obs, info = env.reset(seed=42)
```

### Check Dungeon Stats

```python
obs, info = env.reset()

gen = env.generator
print(f"Rooms: {len(gen.grid)}")
print(f"Start: {gen.start_pos}")
print(f"Exit: {gen.end_pos}")

# Manhattan distance
dist = abs(gen.end_pos[0] - gen.start_pos[0]) + abs(gen.end_pos[1] - gen.start_pos[1])
print(f"Distance: {dist}")
```

### Curriculum Learning

```python
configs = [
    {'min_dist': 3, 'mean_rooms': 5, 'puzzle_density': 0.0},
    {'min_dist': 5, 'mean_rooms': 10, 'puzzle_density': 0.2},
    {'min_dist': 10, 'mean_rooms': 20, 'puzzle_density': 0.4},
]

for i, config in enumerate(configs):
    print(f"Training phase {i+1}")
    env = BoltCrypt(generator_config=config)

    # Train agent on this difficulty
    train(agent, env, episodes=1000)
```

### Benchmark Performance

```bash
python -m boltcrypt.examples.speed_test
```

Expected output:
```
Running benchmarks...
Resets per second: 12000.00
Steps per second: 54000.00
```

---

## Visualization

### Matplotlib Rendering

```python
import matplotlib.pyplot as plt
import numpy as np

env = BoltCrypt()
obs, info = env.reset()

# Create color map
grid = obs['grid']
agent_y, agent_x = obs['agent_pos']

# Mark agent position
display_grid = grid.copy()
display_grid[agent_y, agent_x] = 9  # Special value for agent

plt.imshow(display_grid, cmap='tab10', vmin=0, vmax=9)
plt.colorbar(label='Tile Type')
plt.title('BoltCrypt Room')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Save Dungeon Layout

```python
import pickle

env = BoltCrypt()
obs, info = env.reset()

# Save generator state
with open('dungeon.pkl', 'wb') as f:
    pickle.dump(env.generator, f)

# Load later
with open('dungeon.pkl', 'rb') as f:
    generator = pickle.load(f)
```

---

## Troubleshooting

### "Module not found" error

```bash
# Make sure you're in the BoltCrypt directory
pip install -e .
```

### Episode never ends

Some configurations can create very large dungeons. Add a step limit:

```python
from gymnasium.wrappers import TimeLimit

env = BoltCrypt()
env = TimeLimit(env, max_episode_steps=1000)
```

### Agent stuck in room

Check if puzzle is blocking:

```python
obs, info = env.reset()
room = env.curr_room

print(f"Puzzle: {room.puzzle_type}")
print(f"Solved: {room.is_solved}")
print(f"Doors: {room.doors}")
```

### Low FPS in Pygame

The Pygame visualizer is for debugging, not production. For training, use the base environment without rendering.

---

## Next Steps

- **Read the guides:**
  - [Environment Guide](environment_guide.md) - Deep dive into mechanics
  - [Wrappers Guide](wrappers_guide.md) - Modifying observations and rewards
  - [Puzzle Mechanics](puzzle_mechanics.md) - Understanding each puzzle type
  - [API Reference](api_reference.md) - Complete API documentation

- **Explore examples:**
  - `boltcrypt/examples/tabular_q.py` - Tabular Q-learning
  - `boltcrypt/examples/speed_test.py` - Performance benchmarks
  - `boltcrypt/game/boltcrypt_game.py` - Pygame visualization
  - `boltcrypt/game/boltcrypt_cli.py` - Terminal interface

- **Experiment:**
  - Try different configurations
  - Stack multiple wrappers
  - Create custom puzzle combinations
  - Build your own wrappers

- **Research:**
  - Test exploration methods (RND, ICM)
  - Evaluate memory architectures (LSTM vs MLP)
  - Study generalization across dungeons
  - Benchmark curiosity-driven learning

---

## Getting Help

- **Issues:** Report bugs at [GitHub Issues](https://github.com/yourusername/BoltCrypt/issues)
- **Documentation:** Check the `docs/` folder
- **Examples:** See `boltcrypt/examples/`

Happy dungeon crawling! 🗝️🏹
