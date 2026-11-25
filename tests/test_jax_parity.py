"""
Step-by-step parity between the Python BoltCrypt env and the JAX port on banked levels.

Each config runs every level under the same random action sequence in both implementations and
requires identical observations, rewards, termination and trap flags until the first termination.
Coverage counters make sure the mechanics were actually exercised, so a pass is meaningful.
"""
import os
from collections import Counter

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from boltcrypt.envs.boltcrypt import BoltCrypt  # noqa: E402
from boltcrypt.jaxenv import env as jenv  # noqa: E402
from boltcrypt.jaxenv.level_bank import build_bank  # noqa: E402

# Scale up for a thorough sweep, e.g. BOLTCRYPT_PARITY_LEVELS=2000 BOLTCRYPT_PARITY_STEPS=600
N_LEVELS = int(os.environ.get("BOLTCRYPT_PARITY_LEVELS", 200))
N_STEPS = int(os.environ.get("BOLTCRYPT_PARITY_STEPS", 300))

CONFIGS = {
    "boulders_keys_traps": dict(min_dist=2, mean_rooms=6, std_rooms=1, connectivity=0.4,
                                puzzle_density=0.8, allowed_puzzles=["boulder", "mapped_plates"],
                                key_puzzle_prob=1.0, trap_prob=0.3, min_room_dim=5, max_room_dim=8),
    "stones_warps": dict(min_dist=2, mean_rooms=6, std_rooms=1, connectivity=0.4, puzzle_density=0.8,
                         allowed_puzzles=["stone", "warp_cycle", "warp_rnd"], min_room_dim=4, max_room_dim=7),
    "everything": dict(min_dist=1, mean_rooms=5, std_rooms=1, connectivity=0.6, puzzle_density=0.7,
                       key_puzzle_prob=0.5, trap_prob=0.2, puzzle_required=True, min_room_dim=4,
                       max_room_dim=10),
}


def _python_rollout(cfg, seed, actions):
    env = BoltCrypt(generator_config=cfg)
    obs, _ = env.reset(seed=seed)
    frames, cov = [], Counter()
    for a in actions:
        room, pos = env.curr_room, obs["agent_pos"].copy()
        boulders, stones = [list(b) for b in room.boulders], [list(x) for x in room.stones]
        obs, reward, done, _, info = env.step(int(a))
        frames.append((obs["grid"].copy(), obs["agent_pos"].copy(), obs["global_pos"].copy(),
                       int(obs["inventory"]), np.float32(reward), done, info["trapped"]))
        if env.curr_room is not room:
            cov["door"] += 1
        elif int(np.abs(obs["agent_pos"] - pos).sum()) > 1:
            cov["warp"] += 1
        cov["push"] += room.boulders != boulders
        cov["stone"] += room.stones != stones
        cov["bonus"] += reward == 1.0
        cov["trapped"] += info["trapped"]
        cov["exit"] += done
        if done:
            break
    return frames, cov


def _jax_rollout(bank, actions):
    dev = jenv.bank_to_device(bank)

    def run(level, acts):
        def body(s, a):
            s, r, d, _, trapped = jenv.step(dev, s, a)
            o = jenv.observe(dev, s)
            return s, (o["grid"], o["agent_pos"], o["global_pos"], o["inventory"], r, d, trapped)
        return jax.lax.scan(body, jenv.reset(dev, level), acts)[1]

    out = jax.jit(jax.vmap(run))(jnp.arange(actions.shape[0]), jnp.asarray(actions, jnp.int32))
    return [np.asarray(x) for x in out]


@pytest.mark.parametrize("name", CONFIGS)
def test_jax_matches_python(name):
    cfg = CONFIGS[name]
    seeds = range(1000, 1000 + N_LEVELS)
    bank = build_bank(cfg, seeds)
    actions = np.random.default_rng(0).integers(0, 4, (N_LEVELS, N_STEPS))
    jx = _jax_rollout(bank, actions)

    coverage = Counter()
    fields = ("grid", "agent_pos", "global_pos", "inventory", "reward", "done", "trapped")
    for i, seed in enumerate(seeds):
        frames, cov = _python_rollout(cfg, seed, actions[i])
        coverage += cov
        for t, frame in enumerate(frames):
            for f, py_val, jx_arr in zip(fields, frame, jx):
                if not np.array_equal(py_val, jx_arr[i, t]):
                    pytest.fail(f"{name}: level seed {seed}, step {t}, action {actions[i, t]}: "
                                f"{f} python={py_val!r} jax={jx_arr[i, t]!r}")

    needed = {"boulders_keys_traps": ("door", "push", "bonus", "trapped"),
              "stones_warps": ("door", "warp", "stone"),
              "everything": ("door", "exit")}[name]
    missing = [k for k in needed if coverage[k] == 0]
    assert not missing, f"{name}: mechanics never exercised: {missing} (coverage {dict(coverage)})"
