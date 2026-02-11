import gymnasium as gym

class RoomDiscoveryReward(gym.Wrapper):
    """
    Reward wrapper that grants bonus rewards for discovering new rooms.

    Useful for combating sparse rewards in large dungeons by encouraging exploration
    of the dungeon graph before finding the exit.
    """
    def __init__(self, env, discovery_reward=0.1):
        """
        Initialize room discovery reward wrapper.

        Args:
            env: BoltCrypt environment instance to wrap
            discovery_reward: Bonus reward granted when entering a new room (default 0.1)
        """
        super().__init__(env)
        self.discovery_reward = discovery_reward
        self.visited_rooms = set()

    def reset(self, **kwargs):
        """
        Reset environment and clear visited rooms set.

        Args:
            **kwargs: Arguments passed to base environment reset

        Returns:
            tuple: (observation, info dict)
        """
        obs, info = self.env.reset(**kwargs)
        start_pos = tuple(obs['global_pos'])
        self.visited_rooms = {start_pos}
        return obs, info

    def step(self, action):
        """
        Execute action and add discovery bonus if entering a new room.

        Args:
            action: Action to execute in base environment

        Returns:
            tuple: (observation, modified_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_pos = tuple(obs['global_pos'])

        if curr_pos not in self.visited_rooms:
            reward += self.discovery_reward
            self.visited_rooms.add(curr_pos)

        return obs, reward, terminated, truncated, info