"""
Throughput of the JAX port: batched random-action rollouts with auto-reset and observations.

python -m boltcrypt.examples.jax_speed_test
"""
import time

import jax
import jax.numpy as jnp

from boltcrypt.jaxenv import env as jenv
from boltcrypt.jaxenv.level_bank import build_bank

CONFIG = dict(min_dist=5, mean_rooms=15, std_rooms=2, connectivity=0.3, puzzle_density=0.3,
              key_puzzle_prob=0.5, trap_prob=0.2)
STEPS = 1000


def make_rollout(bank, n_envs):
    @jax.jit
    def rollout(key):
        key, k_lvl = jax.random.split(key)
        states = jax.vmap(lambda lvl: jenv.reset(bank, lvl))(
            jax.random.randint(k_lvl, (n_envs,), 0, bank["seed"].shape[0]))

        def body(carry, _):
            states, key = carry
            key, k_act, k_reset = jax.random.split(key, 3)
            actions = jax.random.randint(k_act, (n_envs,), 0, 4)
            states, reward, done, trunc, _ = jax.vmap(
                lambda s, a, k: jenv.step_autoreset(bank, s, a, k))(
                states, actions, jax.random.split(k_reset, n_envs))
            obs = jax.vmap(lambda s: jenv.observe(bank, s))(states)
            return (states, key), (obs["grid"].sum(), done.sum())

        _, (checksum, dones) = jax.lax.scan(body, (states, key), None, length=STEPS)
        return checksum.sum(), dones.sum()
    return rollout


def main():
    t0 = time.time()
    bank = jenv.bank_to_device(build_bank(CONFIG, range(10_000)))
    print(f"bank: 10,000 levels in {time.time() - t0:.1f}s on {jax.devices()[0]}")
    for n_envs in (1024, 4096, 16384, 65536):
        rollout = make_rollout(bank, n_envs)
        jax.block_until_ready(rollout(jax.random.PRNGKey(0)))  # compile
        t0 = time.time()
        _, dones = jax.block_until_ready(rollout(jax.random.PRNGKey(1)))
        dt = time.time() - t0
        print(f"{n_envs:>6} envs: {n_envs * STEPS / dt / 1e6:6.2f}M steps/s  ({int(dones)} episodes finished)")


if __name__ == "__main__":
    main()
