import gymnasium as gym
import numpy as np

from boltcrypt.environ.boltcrypt import TILE_WALL

class FogOfWar(gym.ObservationWrapper):
    """
    Observation wrapper that limits agent vision to a local window around its position.

    Replaces full room grid with a (2*vision_range + 1) x (2*vision_range + 1) window
    centered on the agent, creating partial observability.
    """
    def __init__(self, env, vision_range=1):
        """
        Initialize fog of war wrapper.

        Args:
            env: BoltCrypt environment instance to wrap
            vision_range: Tiles visible in each direction from agent (default 1 = 3x3 window)
        """
        super().__init__(env)
        self.vision_range = vision_range

    def observation(self, obs):
        """
        Transform observation to include only local vision window.

        Args:
            obs: Full observation dict from base environment

        Returns:
            dict: Modified observation with limited 'grid' field
        """
        grid = obs['grid']
        lx, ly = obs['agent_pos']

        # Pad grid to handle edges
        padded = np.pad(grid, self.vision_range, constant_values=TILE_WALL)

        # Slice the window (adjusting for padding)
        v = self.vision_range
        window = padded[ly:ly + 2*v + 1, lx:lx + 2*v + 1]

        obs['grid'] = window
        return obs