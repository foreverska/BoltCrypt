import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import boltcrypt
import gymnasium as gym


class DQN(nn.Module):
    """Simple feedforward DQN for flattened observations."""

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


def flatten_obs(obs):
    """Flatten the dict observation into a single vector."""
    grid = obs['grid'].flatten()  # 10x10 = 100
    agent_pos = obs['agent_pos']  # 2
    global_pos = obs['global_pos']  # 2
    inventory = np.array([obs['inventory']])  # 1

    return np.concatenate([grid, agent_pos, global_pos, inventory])


def train_dqn(
    episodes=1000,
    max_steps=500,
    batch_size=64,
    gamma=0.99,
    epsilon_start=0.5,
    epsilon_end=0.01,
    teps=3,
    epsilon_decay=0.995,
    learning_rate=1e-4,
    target_update=10,
    buffer_size=100000
):
    """Train DQN agent on BoltCrypt environment."""

    # Create environment
    env = gym.make("BoltCrypt-v0", render_mode="human", max_episode_steps=1000, generator_config={
        'min_dist': 4,
        'mean_rooms': 6,
        'std_rooms': 0,
        'puzzle_density': 0,
        'key_puzzle_prob': 0,
        'min_room_dim': 4,
        'max_room_dim': 10,
        'puzzle_required': False,
        'allowed_puzzles': ['boulder'],
    })

    # Calculate input dimension (grid + agent_pos + global_pos + inventory)
    input_dim = 10 * 10 + 2 + 2 + 1  # 105
    output_dim = env.action_space.n  # 4

    # Initialize networks
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    eps_steps = 0
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=42)
        state = flatten_obs(obs)
        total_reward = 0
        done = False
        truncated = False

        while not done or truncated:
            # Epsilon-greedy action selection
            if eps_steps > 0 or random.random() < epsilon:
                if eps_steps == 0:
                    eps_steps = teps
                else:
                    eps_steps -= 1
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = flatten_obs(next_obs)

            env.render()

            # Store transition
            replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            total_reward += reward

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute Q(s, a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target: r + gamma * max_a' Q_target(s', a')
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done or truncated:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

    return policy_net, episode_rewards


if __name__ == "__main__":
    print("Training DQN on BoltCrypt...")
    policy_net, rewards = train_dqn(episodes=250)

    print("\nTraining complete!")
    print(f"Final 10-episode average reward: {np.mean(rewards[-10:]):.2f}")
