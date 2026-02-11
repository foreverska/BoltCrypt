# BoltCrypt: Procedural Dungeon RL Environment #
BoltCrypt is a lightweight, OpenAI Gymnasium-compatible environment featuring procedurally generated dungeons. It challenges Reinforcement Learning agents (and humans!) to navigate complex layouts, solve sokoban-style boulder puzzles, and manage inventory items like keys to reach the exit.
## 🏰 Features ##
Procedural Generation: Every reset generates a unique dungeon layout based on configurable parameters (density, connectivity, room size).
Puzzle Mechanics: Includes boulder-pushing puzzles and locked doors that require finding a key.
Gymnasium API: Fully compatible with standard RL workflows.
Pygame Visualization: A built-in harness to play manually or watch your agent learn in real-time.
Flexible Observation Space: Provides local room grids, global coordinates, and inventory status.
🛠 Installation
Since this project uses condavenv, ensure you have your environment active:
``` bash
# Example if using conda directly
conda activate <your-env-name>
pip install gymnasium pygame numpy matplotlib boltcrypt
```

## 🚀 Getting Started ##
### Play Manually ###
Test the dungeon generation and mechanics yourself using the Pygame harness:
``` bash
python -m boltcrypt.game.boltcrypt_game
```

**Arrows:** Move the agent.
**R:** Reset/Regenerate the dungeon.
**Goal:** Find the key (if required) and reach the green Exit tile.

Alternatively, play in a terminal-based CLI interface:
``` bash
python -m boltcrypt.game.boltcrypt_cli
```

**WASD/Arrow Keys:** Move the agent.
**R:** Reset/Regenerate the dungeon.

### Train an Agent ###
The project includes a tabular Q-Learning implementation to demonstrate how an agent can "memorize" a specific dungeon layout:
``` bash
python -m boltcrypt.examples.tabular_q
```

### Benchmark Performance ###
Test environment speed with the included benchmark:
``` bash
python -m boltcrypt.examples.speed_test
```

## ⚙️ Configuration ##

The DungeonGenerator and BoltCrypt environment can be customized via a config dictionary:  

### Parameter Description ###
**min_dist:** Minimum Manhattan distance between Start and Exit.
**mean_rooms:** Average number of rooms in the dungeon.
**connectivity:** Probability of creating loops between rooms (0.0 = Tree, 1.0 = Highly connected).
**puzzle_density:** Chance of a room containing a boulder puzzle.  
**key_puzzle_prob:** Chance that the exit is locked and a key is hidden in a leaf room.  
**randomize_end_distance:** If True, randomizes the actual distance from min_dist to 2*min_dist.  


### 🗺 Tile Legend ###
⬜ Floor: Walkable space.  
⬛ Wall: Impassable.  
🚪 Door: Transitions between rooms (may be locked by puzzles).  
🟩 Exit: Your goal!  
🔴 Switch: Target for boulders.  
🟤 Boulder: Can be pushed onto switches.  
🟡 Key: Required to open locked exit rooms.  


### 🤖 Observation Space ###
The environment returns a dictionary:  
**grid:** A 10x10 local view of the current room.  
**agent_pos:** (x, y) coordinates within the room.  
**global_pos:** (gx, gy) coordinates in the dungeon layout.  
**inventory:** Binary flag (1 if holding a key)  

## 📦 Wrappers & Extensions ##
BoltCrypt includes several gym.Wrapper implementations to modify observations or rewards, making it a versatile 
testbed for different RL paradigms.

### 📝 Natural Language Wrapper (NaturalLanguage) ####
The crown jewel for testing Reasoning LLMs. This wrapper transforms the numeric observation space into a rich, 
descriptive narrative. Instead of a grid, the agent receives a text-based description of its surroundings.  
**Dynamic Narrative:** Provides room dimensions, relative positions of doors, boulder locations, and puzzle statuses 
(e.g., "A loud mechanical clank echoes! The doors unlock.").  
**LLM Ready:** Accepts string inputs like "NORTH", "SOUTH", "EAST", or "WEST" in the step() function.  
**Physics Logic:** Includes an "Adventurer's Manual" to explain game rules to an LLM via the observation stream.  

### 🌫️ Fog of War (FogOfWar) ###
Transforms the global room view into a partially observable environment.  
**Vision Range:** Limits the grid observation to a (2v+1) \times (2v+1) window centered on the agent.  
**Memory Challenge:** Forces agents to map the room internally rather than having perfect spatial information.  

### 🏆 Room Discovery Reward (RoomDiscoveryReward) ###
Combats sparse rewards in large dungeons by incentivizing exploration.  
**Exploration Bonus:** Grants a small configurable reward (e.g., +0.1) the first time the agent enters a new room in the dungeon.  
**Global Navigation:** Helps agents learn the layout of the "macro-dungeon" before they’ve found the final exit.  

🛠 Usage Example  
You can stack wrappers to create complex experimental setups:
``` python
import gymnasium as gym
from boltcrypt.env import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage, RoomDiscoveryReward

env = BoltCrypt()
env = RoomDiscoveryReward(env, discovery_reward=0.5)
env = NaturalLanguage(env)

# Now the agent receives text and extra rewards for exploration!
obs, info = env.reset()
print(obs) 

action = "NORTH"
obs, reward, done, trunc, info = env.step(action)
```


Happy Dungeon Crawling! 🗝️🏹