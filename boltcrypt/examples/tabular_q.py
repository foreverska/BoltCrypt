import boltcrypt
import gymnasium as gym

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict


def train_tabular_agent():
    config = {
        'min_dist': 5,  # Small dungeon for fast learning
        'mean_rooms': 10,  # ~10 Rooms total
        'std_rooms': 0,  # Fixed size
        'puzzle_density': 0.3,
        'connectivity': 0.2,  # Mostly tree-like, few loops
        'min_room_dim': 5,
        'max_room_dim': 8,
        'puzzle_required': True,
        'allowed_puzzles': ['boulder']
    }

    env = gym.make('BoltCrypt-v0', render_mode='human', generator_config=config)

    # 2. Hyperparameters
    num_episodes = 2000
    learning_rate = 0.1
    discount_factor = 0.99

    # Exploration (Epsilon Greedy)
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.05

    # The Q-Table
    # Key: (global_x, global_y, local_x, local_y)
    # Value: [Q_north, Q_south, Q_east, Q_west]
    q_table = defaultdict(lambda: np.zeros(4))

    episode_rewards = []
    episode_lengths = []

    print("--- Starting Training ---")
    print(f"Map Config: Min Dist {config['min_dist']}, Total Rooms ~{config['mean_rooms']}")

    for episode in range(num_episodes):
        # Reset with FIXED SEED for every episode
        # This ensures the map layout (Walls/Doors) never changes,
        # allowing the agent to memorize the route.
        obs, _ = env.reset(seed=42)

        gx, gy = obs['global_pos']
        lx, ly = obs['agent_pos']
        state = (gx, gy, lx, ly)

        total_reward = 0
        done = False
        steps = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_obs, reward, done, trunc, _ = env.step(action)

            next_gx, next_gy = next_obs['global_pos']
            next_lx, next_ly = next_obs['agent_pos']
            next_state = (next_gx, next_gy, next_lx, next_ly)

            best_next_q = np.max(q_table[next_state])
            current_q = q_table[state][action]

            q_table[state][action] = current_q + learning_rate * (reward + discount_factor * best_next_q - current_q)

            state = next_state
            total_reward += reward
            steps += 1

            if trunc: break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(episode_rewards[-50:])
            avg_len = np.mean(episode_lengths[-50:])
            print(
                f"Episode {episode + 1:03d} | Avg Reward: {avg_rew:6.2f} | Avg Steps: {avg_len:6.1f} | Epsilon: {epsilon:.2f}")

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    rewards, lengths = train_tabular_agent()