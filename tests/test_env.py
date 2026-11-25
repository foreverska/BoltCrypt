import numpy as np
import pytest

from boltcrypt.envs.boltcrypt import BoltCrypt
from boltcrypt.envs.defines import PuzzleType
from boltcrypt.envs.dungeon_generator import DungeonGenerator
from boltcrypt.envs.step_noise import mix32, step_noise
from boltcrypt.wrapper import NaturalLanguage, RoomDiscoveryReward


def _generate(seed, **cfg):
    g = DungeonGenerator(**cfg)
    g.seed(seed)
    g.generate()
    return g


def _extra_edges(g):
    """Edges beyond a spanning tree, i.e. the loops added by connectivity."""
    return sum(len(r.doors) for r in g.grid.values()) // 2 - (len(g.grid) - 1)


def _rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    frames = [obs]
    for a in actions:
        obs, *_ = env.step(a)
        frames.append(obs)
    return frames


def _same_obs(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ("grid", "agent_pos", "global_pos")) \
        and a["inventory"] == b["inventory"]


# --- Generator ---

def test_connectivity_adds_more_loops_as_it_increases():
    cfg = dict(min_dist=5, mean_rooms=15, std_rooms=0)
    mean_loops = {c: np.mean([_extra_edges(_generate(s, connectivity=c, **cfg)) for s in range(100)])
                  for c in (0.0, 0.5, 1.0)}
    assert mean_loops[0.0] == 0
    assert mean_loops[0.0] < mean_loops[0.5] < mean_loops[1.0]


def test_puzzle_required_does_not_stack_puzzles():
    cfg = dict(min_dist=6, mean_rooms=6, std_rooms=0, puzzle_density=1.0, puzzle_required=True)
    for seed in range(200):
        for room in _generate(seed, **cfg).grid.values():
            pt = room.puzzle_type
            assert not room.stones or pt == PuzzleType.STONE
            assert not room.boulders or pt in (PuzzleType.BOULDER, PuzzleType.BOULDER_PLATES)
            assert not room.warps or pt in (PuzzleType.WARP_CYCLE, PuzzleType.WARP_RND)
            assert len(room.boulders) <= 3


# --- Step randomness ---

def _np_mix32(x):
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def test_step_noise_golden_values():
    # Pinned: any port (e.g. JAX) must reproduce these exactly
    assert [mix32(x) for x in (0, 1, 0xFFFFFFFF, 123456789)] == [0, 1753845952, 1734902346, 2834422664]
    assert [step_noise(*k) for k in [(0, 0, 1, 0), (42, 7, 2, 0), (2**31 - 2, 999, 1, 2), (2**40 + 5, 3, 1, 1)]] \
        == [1492470133, 1012867815, 936945552, 2184228144]


def test_step_noise_matches_uint32_implementation():
    rng = np.random.default_rng(0)
    seeds, steps, idx = (rng.integers(0, 2**32, 2000, dtype=np.uint64) for _ in range(3))
    with np.errstate(over="ignore"):
        h = _np_mix32(seeds)
        h = _np_mix32(h ^ steps.astype(np.uint32))
        h = _np_mix32(h ^ np.uint32(2))
        h = _np_mix32(h ^ idx.astype(np.uint32))
    expected = [step_noise(int(s), int(t), 2, int(i)) for s, t, i in zip(seeds, steps, idx)]
    assert h.tolist() == expected


# --- Environment determinism ---

STOCHASTIC_CFG = dict(min_dist=3, mean_rooms=6, std_rooms=0, puzzle_density=1.0,
                      allowed_puzzles=["stone", "warp_rnd"], min_room_dim=7)


@pytest.mark.parametrize("seed", range(10))
def test_trajectory_depends_only_on_seed_and_actions(seed):
    actions = np.random.default_rng(seed).integers(0, 4, 400).tolist()
    env_a, env_b, other = (BoltCrypt(generator_config=STOCHASTIC_CFG) for _ in range(3))

    first = _rollout(env_a, seed, actions)
    # Interleave unrelated stepping so any hidden shared RNG state would show up
    other.reset(seed=seed + 1000)
    for a in actions[:50]:
        other.step(a)
    second = _rollout(env_b, seed, actions)

    assert all(_same_obs(x, y) for x, y in zip(first, second))


def test_stochastic_dynamics_still_happen():
    """Stones move and random warps relocate the agent; the refactor must not freeze them."""
    env = BoltCrypt(generator_config=STOCHASTIC_CFG)
    stone_moves = warps = 0
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        for a in np.random.default_rng(seed).integers(0, 4, 400):
            stones_before = [list(s) for s in env.curr_room.stones]
            room_before, pos_before = env.curr_room, obs["agent_pos"].copy()
            obs, *_ = env.step(int(a))
            if env.curr_room is room_before:
                stone_moves += env.curr_room.stones != stones_before
                warps += int(np.abs(obs["agent_pos"] - pos_before).sum()) > 1
    assert stone_moves > 0 and warps > 0


def test_unseeded_resets_follow_first_seed():
    def levels(env):
        env.reset(seed=5)
        return [env.reset()[0]["grid"].tobytes() + str(env.generator.end_pos).encode() for _ in range(5)]
    assert levels(BoltCrypt()) == levels(BoltCrypt())


# --- Natural language wrapper ---

KEY_CFG = dict(min_dist=2, mean_rooms=5, std_rooms=0, key_puzzle_prob=1.0)


def test_natlang_stacks_on_other_wrappers():
    env = NaturalLanguage(RoomDiscoveryReward(BoltCrypt(generator_config=KEY_CFG)))
    text, _ = env.reset(seed=0)
    assert "SITUATION REPORT" in text
    env.step("north, east, look, hint")


@pytest.mark.parametrize("seed", range(10))
def test_natlang_describes_key_locked_exit_room(seed):
    env = NaturalLanguage(BoltCrypt(generator_config=KEY_CFG))
    env.reset(seed=seed)
    base = env.unwrapped
    assert base.generator.grid[base.generator.end_pos].is_locked

    # Without the key, a door into the locked exit is reported as needing one
    base.gx, base.gy = next(p for p, r in base.generator.grid.items()
                            if any((p[0] + d.value[0], p[1] + d.value[1]) == base.generator.end_pos
                                   for d in r.doors))
    base.curr_room = base.generator.grid[(base.gx, base.gy)]
    env.just_entered_room = True
    assert "NEEDS KEY" in env._generate_text(base._get_obs(), 0, False, "")

    # Inside the locked exit room (entered with the key) the description must not crash
    base.inventory_has_key = True
    base.gx, base.gy = base.generator.end_pos
    base.curr_room = base.generator.grid[base.generator.end_pos]
    env.just_entered_room = True
    assert "ARCHWAY" in env._generate_text(base._get_obs(), 0, False, "")


def test_natlang_text_is_reproducible():
    commands = ["look", "hint", "north", "east", "south", "west"] * 20

    def transcript():
        env = NaturalLanguage(BoltCrypt(generator_config=dict(puzzle_density=1.0, min_room_dim=7)))
        out = [env.reset(seed=3)[0]]
        out += [env.step(c)[0] for c in commands]
        return out

    assert transcript() == transcript()
