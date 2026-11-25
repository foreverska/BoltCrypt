from collections import deque

import numpy as np
import pytest

from boltcrypt.envs.boltcrypt import BoltCrypt
from boltcrypt.envs.defines import TILE_EMPTY, TILE_KEY
from boltcrypt.wrapper import NaturalLanguage

# No puzzles, so the full state is (room, tile, has_key) and can be searched exhaustively
TRAP_CFG = dict(min_dist=3, mean_rooms=10, std_rooms=0, connectivity=0.3, key_puzzle_prob=1.0,
                min_room_dim=4, max_room_dim=6, trap_prob=1.0)


def _set_state(env, state):
    gx, gy, lx, ly, has_key = state
    for room in env.generator.grid.values():
        room.sprung = False
        if room.has_key:
            ky, kx = room.key_pos
            room.grid_array[ky, kx] = TILE_EMPTY if has_key else TILE_KEY
    env.gx, env.gy, env.lx, env.ly = gx, gy, lx, ly
    env.curr_room = env.generator.grid[(gx, gy)]
    env.inventory_has_key = has_key


def _exit_reachable(env):
    """BFS through the real step function; stepping into a trap ends that branch."""
    start = (env.gx, env.gy, env.lx, env.ly, False)
    seen, queue = {start}, deque([start])
    while queue:
        state = queue.popleft()
        for a in range(4):
            _set_state(env, state)
            obs, _, done, _, info = env.step(a)
            if done:
                return True
            if info["trapped"]:
                continue
            nxt = (*obs["global_pos"].tolist(), *obs["agent_pos"].tolist(), bool(obs["inventory"]))
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _walk_into_trap(env):
    """Teleport next to a trap's door from a neighbour and step through. Returns False if no trap."""
    grid = env.generator.grid
    for pos, room in grid.items():
        for d in room.doors:
            n_pos = (pos[0] + d.value[0], pos[1] + d.value[1])
            if not grid[n_pos].is_trap or room.puzzle_type.name != "NONE":
                continue
            off = room.doors[d]
            # Stand on the tile just inside the door, facing it
            lx, ly = {"NORTH": (off, room.h - 2), "SOUTH": (off, 1),
                      "EAST": (room.w - 2, off), "WEST": (1, off)}[d.name]
            _set_state(env, (*pos, lx, ly, True))
            action = {"NORTH": 0, "SOUTH": 1, "EAST": 2, "WEST": 3}[d.name]
            _, _, _, _, info = env.step(action)
            assert info["trapped"] and (env.gx, env.gy) == n_pos
            return True
    return False


def test_traps_are_placed():
    env = BoltCrypt(generator_config=TRAP_CFG)
    counts = []
    for seed in range(50):
        env.reset(seed=seed)
        counts.append(sum(r.is_trap for r in env.generator.grid.values()))
    assert np.mean(counts) > 1


def test_critical_rooms_are_never_traps():
    env = BoltCrypt(generator_config=TRAP_CFG)
    for seed in range(50):
        env.reset(seed=seed)
        for room in env.generator.grid.values():
            assert not (room.is_trap and (room.is_start or room.is_exit or room.has_key))


@pytest.mark.parametrize("seed", range(25))
def test_exit_reachable_with_traps(seed):
    env = BoltCrypt(generator_config=TRAP_CFG)
    env.reset(seed=seed)
    assert _exit_reachable(env)


def test_trap_seals_for_rest_of_episode():
    env = BoltCrypt(generator_config=TRAP_CFG)
    for seed in range(20):
        env.reset(seed=seed)
        if _walk_into_trap(env):
            break
    else:
        pytest.fail("no reachable trap found")
    trap_pos = (env.gx, env.gy)
    for a in np.random.default_rng(0).integers(0, 4, 2000):
        obs, _, done, _, info = env.step(int(a))
        assert tuple(obs["global_pos"]) == trap_pos and info["trapped"] and not done


def test_traps_do_not_change_layout():
    """Same seed with and without traps: identical dungeon apart from the trap flags."""
    def layout(trap_prob):
        env = BoltCrypt(generator_config={**TRAP_CFG, "trap_prob": trap_prob, "puzzle_density": 0.5})
        env.reset(seed=7)
        return {pos: (r.w, r.h, dict(r.doors), r.puzzle_type, r.has_key, r.key_pos, r.grid_array.tobytes())
                for pos, r in env.generator.grid.items()}, env.generator.end_pos
    assert layout(0.0) == layout(1.0)


def test_natlang_reports_sealed_doors():
    env = NaturalLanguage(BoltCrypt(generator_config=TRAP_CFG))
    for seed in range(20):
        env.reset(seed=seed)
        if _walk_into_trap(env.unwrapped):
            break
    env.just_entered_room = True
    text = env._generate_text(env.unwrapped._get_obs(), 0, False, "")
    assert "(SEALED)" in text and "(OPEN)" not in text
    assert "sealed" in env.step("hint")[0]
