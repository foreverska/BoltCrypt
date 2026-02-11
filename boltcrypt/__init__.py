from gymnasium.envs.registration import register

register(
    id='BoltCrypt-v0',
    entry_point='boltcrypt.environ:BoltCrypt',
    max_episode_steps=1000
)