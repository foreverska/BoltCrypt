from collections import deque

import numpy as np
import pytest

from boltcrypt.envs.boltcrypt import BoltCrypt
from boltcrypt.envs.defines import PuzzleType, TILE_EMPTY, TILE_KEY, TILE_LAVA
from boltcrypt.wrapper import NaturalLanguage

LAVA_CFG = dict(min_dist=3, mean_rooms=8, std_rooms=0, connectivity=0.3, key_puzzle_prob=1.0,
                min_room_dim=5, max_room_dim=8, lava_density=0.6)


def _set_state(env, state):
    gx, gy, lx, ly, has_key = state
    for room in env.generator.grid.values():
        if room.has_key:
            ky, kx = room.key_pos
            room.grid_array[ky, kx] = TILE_EMPTY if has_key else TILE_KEY
    env.gx, env.gy, env.lx, env.ly = gx, gy, lx, ly
    env.curr_room = env.generator.grid[(gx, gy)]
    env.inventory_has_key = has_key


def _exit_reachable_alive(env):
    """BFS through the real step function; a step into lava ends that branch."""
    start = (env.gx, env.gy, env.lx, env.ly, False)
    seen, queue = {start}, deque([start])
    while queue:
        state = queue.popleft()
        for a in range(4):
            _set_state(env, state)
            obs, _, done, _, info = env.step(a)
            if info["died"]:
                continue
            if done:
                return True
            nxt = (*obs["global_pos"].tolist(), *obs["agent_pos"].tolist(), bool(obs["inventory"]))
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _lava_tiles(env):
    return sum(len(r.lava) for r in env.generator.grid.values())


def test_lava_is_placed_only_in_puzzle_free_rooms():
    env = BoltCrypt(generator_config={**LAVA_CFG, "puzzle_density": 0.5, "min_room_dim": 7})
    total = 0
    for seed in range(30):
        env.reset(seed=seed)
        total += _lava_tiles(env)
        for room in env.generator.grid.values():
            assert not room.lava or room.puzzle_type == PuzzleType.NONE
            for (y, x) in room.lava:
                assert room.grid_array[y, x] == TILE_LAVA
    assert total > 100


@pytest.mark.parametrize("seed", range(25))
def test_exit_reachable_without_dying(seed):
    env = BoltCrypt(generator_config=LAVA_CFG)
    env.reset(seed=seed)
    assert _lava_tiles(env) > 0
    assert _exit_reachable_alive(env)


def test_lava_does_not_change_layout():
    def layout(density):
        env = BoltCrypt(generator_config={**LAVA_CFG, "lava_density": density, "trap_prob": 0.3,
                                          "puzzle_density": 0.5, "min_room_dim": 7})
        env.reset(seed=11)
        return env.generator
    plain, lava = layout(0.0), layout(0.6)
    assert plain.end_pos == lava.end_pos and set(plain.grid) == set(lava.grid)
    for pos, r in plain.grid.items():
        l = lava.grid[pos]
        assert (r.w, r.h, r.doors, r.puzzle_type, r.key_pos, r.is_trap) == (l.w, l.h, l.doors, l.puzzle_type, l.key_pos, l.is_trap)
        differ = np.argwhere(r.grid_array != l.grid_array)
        assert {tuple(d) for d in differ} == l.lava


def test_stepping_on_lava_terminates():
    env = BoltCrypt(generator_config=LAVA_CFG)
    for seed in range(20):
        env.reset(seed=seed)
        room = env.curr_room
        for (y, x) in room.lava:
            for a, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                if room.grid_array[y - dy, x - dx] not in (TILE_LAVA,) and 0 < y - dy < room.h - 1 \
                        and 0 < x - dx < room.w - 1:
                    env.lx, env.ly = x - dx, y - dy
                    _, reward, done, _, info = env.step(a)
                    assert done and info["died"] and reward < 0
                    return
    pytest.fail("no lava next to a walkable tile in the start room")


def test_natlang_reports_death_not_victory():
    env = NaturalLanguage(BoltCrypt(generator_config=LAVA_CFG))
    for seed in range(20):
        env.reset(seed=seed)
        base = env.unwrapped
        room = base.curr_room
        cands = [(y, x) for (y, x) in room.lava if (y - 1, x) not in room.lava and 1 <= y - 1]
        if cands:
            y, x = cands[0]
            base.lx, base.ly = x, y - 1
            text, _, done, _, _ = env.step("north")
            assert done and "PERISHED" in text and "VICTORY" not in text
            return
    pytest.fail("no lava reachable from below in a start room")
