# BoltCrypt Documentation

Welcome to the BoltCrypt documentation! This guide will help you understand and use the BoltCrypt reinforcement learning environment.

---

## 📚 Documentation Index

### [Quick Start Guide](quickstart.md)
Get up and running with BoltCrypt in minutes. Covers installation, basic usage, and simple examples.

**Best for:** First-time users, quick reference

---

### [Environment Guide](environment_guide.md)
Comprehensive guide to the BoltCrypt environment, including:
- Room-based dungeon structure
- Observation and action spaces
- Dungeon generation pipeline
- Episode structure and mechanics
- Reward systems
- Difficulty tuning
- Performance characteristics

**Best for:** Understanding core mechanics, configuring dungeons, optimization

---

### [Wrappers Guide](wrappers_guide.md)
Complete reference for environment wrappers:
- **NaturalLanguage** - Text descriptions for LLM agents
- **FogOfWar** - Partial observability
- **RoomDiscoveryReward** - Exploration bonuses
- Creating custom wrappers

**Best for:** Modifying observations, reward shaping, LLM integration

---

### [Puzzle Mechanics](puzzle_mechanics.md)
Deep dive into the five puzzle types:
- **BOULDER** - Classic Sokoban pushing puzzles
- **BOULDER_PLATES** - State-dependent door mappings
- **STONE** - Random moving obstacles (RND killer)
- **WARP_CYCLE** - Hamilton cycle teleporters (ICM killer)
- **WARP_RND** - Chaotic random teleportation (ICM++ killer)

**Best for:** Research design, understanding challenges, debugging puzzles

---

### [API Reference](api_reference.md)
Complete API documentation for all classes and methods:
- `BoltCrypt` environment
- `DungeonGenerator`
- `Room` class
- All wrappers
- Enums and constants

**Best for:** API lookup, development, detailed specifications

---

## 🚀 Quick Links

### Getting Started
1. [Install BoltCrypt](quickstart.md#installation)
2. [Run your first episode](quickstart.md#basic-usage)
3. [Play manually](quickstart.md#play-manually)
4. [Configure a dungeon](quickstart.md#configuration-examples)

### Common Tasks
- [Use wrappers](quickstart.md#using-wrappers)
- [Train an agent](quickstart.md#training-a-simple-agent)
- [Tune difficulty](environment_guide.md#difficulty-tuning)
- [Benchmark performance](quickstart.md#benchmark-performance)

### Research Applications
- [Test curiosity methods](puzzle_mechanics.md#testing-curiosity-methods)
- [Evaluate memory architectures](puzzle_mechanics.md#testing-memory)
- [Study generalization](puzzle_mechanics.md#testing-generalization)
- [LLM integration](wrappers_guide.md#naturallanguage)

---

## 📖 Reading Guide

### For New Users
1. Start with [Quick Start](quickstart.md)
2. Read [Environment Guide](environment_guide.md) sections:
   - Core Concepts
   - Observation Space
   - Movement Mechanics
3. Try the manual play interfaces
4. Experiment with configurations

### For Researchers
1. Skim [Quick Start](quickstart.md)
2. Read [Puzzle Mechanics](puzzle_mechanics.md) to understand challenges
3. Review [Environment Guide](environment_guide.md) for configuration
4. Use [Wrappers Guide](wrappers_guide.md) for modifications
5. Reference [API Reference](api_reference.md) as needed

### For Developers
1. Read [API Reference](api_reference.md)
2. Study [Environment Guide](environment_guide.md) - Episode Structure
3. Review [Wrappers Guide](wrappers_guide.md) - Creating Custom Wrappers
4. Check source code docstrings

---

## 🎯 Use Cases

### Reinforcement Learning Research

**Exploration Methods:**
```python
# Test RND on stochastic puzzles
env = BoltCrypt({'allowed_puzzles': ['stone', 'warp_rnd']})
```

**Memory Architectures:**
```python
# Require memory with partial observability
env = FogOfWar(BoltCrypt(), vision_range=1)
```

**Curriculum Learning:**
```python
# Progressive difficulty
for dist in [3, 5, 8, 12]:
    env = BoltCrypt({'min_dist': dist})
    train(agent, env)
```

### LLM Agent Testing

**Natural Language Interface:**
```python
env = NaturalLanguage(BoltCrypt())
obs, _ = env.reset()
# obs is rich text description
```

**Planning Challenges:**
```python
config = {'allowed_puzzles': ['boulder', 'mapped_plates']}
env = NaturalLanguage(BoltCrypt(config))
```

### Multi-Agent Research

**Cooperative Navigation:**
```python
# Create multiple agent instances
agents = [Agent() for _ in range(3)]
# Share dungeon but track separate positions
```

### Benchmark Environment

**Standardized Testing:**
```python
# Fixed seed for reproducibility
env = BoltCrypt()
for seed in range(100):
    obs, _ = env.reset(seed=seed)
    evaluate(agent, env)
```

---

## 🔬 Research Questions

BoltCrypt is designed to address:

### Exploration
- Do curiosity-driven methods handle stochastic environments?
- How do agents explore sparse-reward dungeons?
- Does room discovery shaping improve learning?

### Memory
- Can memoryless agents solve teleporter puzzles?
- How much memory is needed for navigation?
- Do recurrent architectures generalize better?

### Generalization
- Do agents transfer puzzle-solving skills across dungeons?
- How does connectivity affect learning difficulty?
- Can agents adapt to new puzzle types?

### Planning
- How far ahead do agents plan in boulder puzzles?
- Do LLMs leverage spatial reasoning?
- Can agents solve multi-step navigation problems?

---

## 📊 Example Configurations

### Minimal (Testing)
```python
{'min_dist': 1, 'mean_rooms': 3, 'puzzle_density': 0.0}
```

### Easy (Learning)
```python
{'min_dist': 3, 'mean_rooms': 8, 'connectivity': 0.8, 'puzzle_density': 0.0}
```

### Medium (Standard)
```python
{'min_dist': 8, 'mean_rooms': 15, 'connectivity': 0.5, 'puzzle_density': 0.3}
```

### Hard (Challenge)
```python
{'min_dist': 15, 'mean_rooms': 40, 'connectivity': 0.1, 'puzzle_density': 0.6}
```

### Extreme (Research)
```python
{'min_dist': 20, 'mean_rooms': 60, 'connectivity': 0.0, 'puzzle_density': 0.9,
 'allowed_puzzles': ['warp_rnd', 'stone'], 'key_puzzle_prob': 1.0}
```

---

## 🛠️ Tools and Utilities

### Manual Play
- **Pygame:** `python -m boltcrypt.game.boltcrypt_game`
- **Terminal:** `python -m boltcrypt.game.boltcrypt_cli`

### Benchmarks
- **Performance:** `python -m boltcrypt.examples.speed_test`

### Training Examples
- **Tabular Q:** `python -m boltcrypt.examples.tabular_q`

---

## 📝 Additional Resources

### Code Examples
- `boltcrypt/examples/` - Training scripts
- `boltcrypt/game/` - Interactive interfaces
- `boltcrypt/wrapper/` - Wrapper implementations

### Source Code
- `boltcrypt/env/boltcrypt.py` - Core environment
- All classes include comprehensive docstrings

### Community
- GitHub Issues - Bug reports and questions
- Discussions - Research ideas and feedback

---

## 🔖 Quick Reference

### Actions
- `0` - North (up)
- `1` - South (down)
- `2` - East (right)
- `3` - West (left)

### Tile Types
- `0` - Empty floor
- `1` - Wall
- `2` - Door
- `3` - Exit
- `4` - Switch
- `5` - Boulder
- `6` - Key
- `7` - Stone
- `8` - Warp

### Default Rewards
- `-0.01` - Time step
- `+1.0` - Key/puzzle
- `+10.0` - Exit

---

## 📧 Support

- **Documentation Issues:** Check all guide sections
- **Bug Reports:** GitHub Issues
- **Feature Requests:** GitHub Discussions
- **Research Questions:** Check Puzzle Mechanics guide

---

Happy exploring! 🗝️🏹
