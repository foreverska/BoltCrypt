"""
Functional JAX port of BoltCrypt.step, bit-exact with the Python environment on banked levels.

Every branch of BoltCrypt.step is mirrored as a masked select, in the same order, so the two can
be checked step by step (tests/test_jax_parity.py). All in-episode randomness comes from the same
counter-based hash as boltcrypt.envs.step_noise. Functions operate on a single environment;
use jax.vmap for batches.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from boltcrypt.envs.defines import (TILE_BOULDER, TILE_COLUMN, TILE_DOOR, TILE_EMPTY, TILE_EXIT,
                                    TILE_KEY, TILE_LAVA, TILE_STONE, TILE_WALL)
from boltcrypt.envs.step_noise import DIRECTION_PERMS, NOISE_STONE, NOISE_WARP

P_BOULDER, P_PLATES, P_WARP_CYCLE, P_WARP_RND = 1, 2, 4, 5

# (dx, dy) per action, matching BoltCrypt.step: 0=N, 1=S, 2=E, 3=W
ACTION_DELTA = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], np.int32)
OPPOSITE = np.array([1, 0, 3, 2], np.int32)
# (dy, dx) stone moves in Room.move_stones order, and their shuffles
STONE_MOVES = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)], np.int32)
PERMS = np.array(DIRECTION_PERMS, np.int32)

BANK_KEYS = ("seed", "start_room", "key_room", "key_pos", "room_xy", "room_w", "room_h", "neighbor",
             "door_off", "grid", "ptype", "is_locked", "is_trap", "switches", "plate_dir",
             "boulders0", "n_boulders", "stones0", "n_stones", "cands", "n_cands", "cand_index")


class State(NamedTuple):
    level: jnp.ndarray       # index into the bank
    room: jnp.ndarray
    lx: jnp.ndarray
    ly: jnp.ndarray
    has_key: jnp.ndarray
    t: jnp.ndarray           # episode step counter (BoltCrypt.episode_step)
    boulders: jnp.ndarray    # [R, 3, 2] (y, x), -1 padded
    stones: jnp.ndarray      # [R, 3, 2] (y, x), -1 padded
    sprung: jnp.ndarray      # [R] trap rooms that have sealed
    solved_once: jnp.ndarray  # [R] Room.previously_solved


class Params(NamedTuple):
    puzzle_bonus: bool = True
    key_bonus: bool = True
    max_steps: int = 1000    # truncation, as gymnasium's TimeLimit on BoltCrypt-v0


# The large per-tile arrays stay compact on device and are widened where they are read
COMPACT_KEYS = ("grid", "cand_index", "cands")


def bank_to_device(bank: dict) -> dict:
    """Keep only the arrays the step function reads; small ones as int32, large ones compact."""
    out = {}
    for k in BANK_KEYS:
        v = np.asarray(bank[k])
        keep = v.dtype in (bool, np.uint32) or k in COMPACT_KEYS
        out[k] = jnp.asarray(v if keep else v.astype(np.int32))
    return out


# --- step_noise in uint32 arithmetic ---

def _mix32(x):
    x = x.astype(jnp.uint32)
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def step_noise(seed, step, purpose, index=0):
    h = _mix32(seed)
    h = _mix32(h ^ jnp.asarray(step).astype(jnp.uint32))
    h = _mix32(h ^ jnp.uint32(purpose))
    return _mix32(h ^ jnp.asarray(index).astype(jnp.uint32))


def _randint(n, seed, step, purpose, index=0):
    return (step_noise(seed, step, purpose, index) % jnp.maximum(n, 1).astype(jnp.uint32)).astype(jnp.int32)


# --- helpers ---

def _at(pos, y, x):
    """Mask of rows in pos [K, 2] equal to (y, x). -1 padding never matches a real tile."""
    return (pos[:, 0] == y) & (pos[:, 1] == x)


def _room_grid(bank, s, room):
    """Static tiles of a room, with the key tile cleared once the key is held."""
    grid = bank["grid"][s.level, room].astype(jnp.int32)
    m = grid.shape[0]
    ys, xs = jnp.arange(m)[:, None], jnp.arange(m)[None, :]
    ky, kx = bank["key_pos"][s.level, 0], bank["key_pos"][s.level, 1]
    taken = s.has_key & (room == bank["key_room"][s.level]) & (ys == ky) & (xs == kx)
    return jnp.where(taken, TILE_EMPTY, grid)


def _tile(grid, y, x):
    m = grid.shape[0]
    return grid[jnp.clip(y, 0, m - 1), jnp.clip(x, 0, m - 1)]


def _is_blocking(tile):
    return (tile == TILE_WALL) | (tile == TILE_COLUMN) | (tile == TILE_DOOR) | (tile == TILE_EXIT)


def _move_stones(bank, s, grid, w, h):
    """Room.move_stones: each stone, in order, takes the first legal move of a hashed shuffle."""
    stones = s.stones[s.room]
    boulders = s.boulders[s.room]
    n = bank["n_stones"][s.level, s.room]
    seed = bank["seed"][s.level]
    for i in range(stones.shape[0]):
        # All four candidates at once, in hashed-shuffle order; take the first legal one.
        # (A chain of per-candidate selects here was miscompiled by XLA:CPU 0.12.1 under vmap,
        # splicing y and x from different candidates. Keep this vectorised form.)
        perm = jnp.asarray(PERMS)[_randint(len(PERMS), seed, s.t, NOISE_STONE, i)]
        cand = stones[i][None, :] + jnp.asarray(STONE_MOVES)[perm]       # [4, (y, x)]
        ny, nx = cand[:, 0], cand[:, 1]
        others = (jnp.arange(stones.shape[0]) != i)[None, :]
        ok = (1 <= ny) & (ny < h - 1) & (1 <= nx) & (nx < w - 1)
        ok &= ~_is_blocking(_tile(grid, ny, nx))
        ok &= ~((ny == s.ly) & (nx == s.lx))
        ok &= ~jnp.any((stones[None, :, 0] == ny[:, None]) & (stones[None, :, 1] == nx[:, None]) & others, axis=1)
        ok &= ~jnp.any((boulders[None, :, 0] == ny[:, None]) & (boulders[None, :, 1] == nx[:, None]), axis=1)
        new = cand[jnp.argmax(ok)]
        stones = stones.at[i].set(jnp.where((i < n) & jnp.any(ok), new, stones[i]))
    return s._replace(stones=s.stones.at[s.room].set(stones))


def _boulder_solved(bank, s, room, boulders):
    """Room.check_solved for BOULDER rooms: every switch covered by a boulder."""
    sw = bank["switches"][s.level, room]
    valid = sw[:, 0] >= 0
    covered = jnp.any((sw[:, None, 0] == boulders[None, :, 0]) & (sw[:, None, 1] == boulders[None, :, 1]), axis=1)
    return jnp.all(covered | ~valid)


# --- public API ---

def reset(bank, level):
    """Start an episode on bank level `level` (BoltCrypt.reset with that level's seed)."""
    room = bank["start_room"][level]
    r_max = bank["grid"].shape[1]
    return State(
        level=jnp.asarray(level, jnp.int32), room=room,
        lx=bank["room_w"][level, room] // 2, ly=bank["room_h"][level, room] // 2,
        has_key=jnp.zeros((), bool), t=jnp.zeros((), jnp.int32),
        boulders=bank["boulders0"][level], stones=bank["stones0"][level],
        sprung=jnp.zeros(r_max, bool), solved_once=jnp.zeros(r_max, bool))


def observe(bank, s):
    """BoltCrypt._get_obs as a dict of arrays."""
    grid = _room_grid(bank, s, s.room)
    m = grid.shape[0]
    ys, xs = jnp.arange(m)[:, None, None], jnp.arange(m)[None, :, None]
    b, st = s.boulders[s.room], s.stones[s.room]
    grid = jnp.where(jnp.any((ys == b[:, 0]) & (xs == b[:, 1]), -1), TILE_BOULDER, grid)
    grid = jnp.where(jnp.any((ys == st[:, 0]) & (xs == st[:, 1]), -1), TILE_STONE, grid)
    return {"grid": grid.astype(jnp.int8),
            "agent_pos": jnp.stack([s.lx, s.ly]),
            "global_pos": bank["room_xy"][s.level, s.room],
            "inventory": s.has_key.astype(jnp.int32)}


def step(bank, s, action, params: Params = Params()):
    """
    BoltCrypt.step for one environment. Returns (state, reward, terminated, truncated, info) with
    info = {"trapped", "died"} as in Python. Call observe() for the observation. Stepping after
    termination is as undefined as in Python.
    """
    lvl, room = s.level, s.room
    seed = bank["seed"][lvl]
    w, h = bank["room_w"][lvl, room], bank["room_h"][lvl, room]
    grid = _room_grid(bank, s, room)
    ptype = bank["ptype"][lvl, room]

    # 0. Sailing stones move first, using the pre-action agent position
    s = _move_stones(bank, s, grid, w, h)
    t = s.t
    s = s._replace(t=t + 1)

    # 1. Parse action
    dx, dy = jnp.asarray(ACTION_DELTA)[action, 0], jnp.asarray(ACTION_DELTA)[action, 1]
    tx, ty = s.lx + dx, s.ly + dy
    reward = jnp.float32(-0.01)
    done = jnp.zeros((), bool)

    in_bounds = (0 <= tx) & (tx < w) & (0 <= ty) & (ty < h)
    static = _tile(grid, ty, tx)
    stones, boulders = s.stones[room], s.boulders[room]
    hit_stone = jnp.any(_at(stones, ty, tx))
    at_boulder = _at(boulders, ty, tx)

    # 2a. Key pickup
    c_key = in_bounds & ~hit_stone & (static == TILE_KEY)
    # 2b. Boulder push
    c_boulder = in_bounds & ~hit_stone & (static != TILE_KEY) & jnp.any(at_boulder)
    px, py = tx + dx, ty + dy
    safe = (2 <= px) & (px <= w - 3) & (2 <= py) & (py <= h - 3)
    blocked = _is_blocking(_tile(grid, py, px)) | jnp.any(_at(boulders, py, px)) | jnp.any(_at(stones, py, px))
    pushed = c_boulder & safe & ~blocked
    new_boulders = jnp.where(at_boulder[:, None] & pushed, jnp.stack([py, px]), boulders)
    solved_now = (ptype == P_BOULDER) & _boulder_solved(bank, s, room, new_boulders)
    already = s.solved_once[room]
    puzzle_reward = pushed & solved_now & ~already & params.puzzle_bonus
    # 2c. Walkable tile (doors handled separately)
    c_walk = in_bounds & ~hit_stone & (static != TILE_KEY) & ~jnp.any(at_boulder) \
        & (static != TILE_WALL) & (static != TILE_COLUMN)
    c_door = c_walk & (static == TILE_DOOR)
    c_floor = c_walk & (static != TILE_DOOR)

    # Door locks: unsolved boulder room, wrong (or no) mapped plate, sprung trap
    locked = (ptype == P_BOULDER) & ~_boulder_solved(bank, s, room, boulders)
    on_plate = _at(bank["switches"][lvl, room], boulders[0, 0], boulders[0, 1])
    plate_dir = jnp.max(jnp.where(on_plate, bank["plate_dir"][lvl, room], -1))
    locked |= (ptype == P_PLATES) & (plate_dir != action)
    locked |= s.sprung[room]
    target = bank["neighbor"][lvl, room, action]
    tgt = jnp.maximum(target, 0)
    key_blocked = bank["is_locked"][lvl, tgt] & ~s.has_key
    transition = c_door & ~locked & (target >= 0) & ~key_blocked

    # Spawn one tile inside the target room's matching door
    off = bank["door_off"][lvl, tgt, jnp.asarray(OPPOSITE)[action]]
    tw, th = bank["room_w"][lvl, tgt], bank["room_h"][lvl, tgt]
    spawn_x = jnp.array([off, off, 1, tw - 2])[action]
    spawn_y = jnp.array([1, th - 2, off, off])[action]

    moved = c_key | pushed | c_floor
    new_room = jnp.where(transition, tgt, room)
    lx = jnp.where(transition, spawn_x, jnp.where(moved, tx, s.lx))
    ly = jnp.where(transition, spawn_y, jnp.where(moved, ty, s.ly))

    reward = jnp.where(c_key & params.key_bonus, 1.0, reward)
    reward = jnp.where(puzzle_reward, 1.0, reward)
    at_exit = c_floor & (static == TILE_EXIT)
    reward = jnp.where(at_exit, 10.0, reward)
    died = c_floor & (static == TILE_LAVA)
    done = at_exit | died

    s = s._replace(
        room=new_room, lx=lx, ly=ly,
        has_key=s.has_key | c_key,
        boulders=s.boulders.at[room].set(new_boulders),
        solved_once=s.solved_once.at[room].set(already | (pushed & solved_now)),
        sprung=s.sprung.at[tgt].set(s.sprung[tgt] | (transition & bank["is_trap"][lvl, tgt])))

    # 3. Warps act on wherever the agent now stands, including after a bump
    room = s.room
    wtype = bank["ptype"][lvl, room]
    ci = bank["cand_index"][lvl, room, s.ly, s.lx].astype(jnp.int32)
    n = bank["n_cands"][lvl, room]
    on_warp = ((wtype == P_WARP_CYCLE) | (wtype == P_WARP_RND)) & (ci >= 0)
    k = _randint(n - 1, seed, t, NOISE_WARP)
    dest_i = jnp.where(wtype == P_WARP_CYCLE, (ci + 1) % jnp.maximum(n, 1), k + (k >= ci))
    dest = bank["cands"][lvl, room, jnp.clip(dest_i, 0, bank["cands"].shape[2] - 1)].astype(jnp.int32)
    warp = on_warp & ~jnp.any(_at(s.stones[room], dest[0], dest[1]))
    s = s._replace(ly=jnp.where(warp, dest[0], s.ly), lx=jnp.where(warp, dest[1], s.lx))

    truncated = (s.t >= params.max_steps) & ~done
    return s, reward, done, truncated, {"trapped": s.sprung[s.room], "died": died}


def step_autoreset(bank, s, action, key, params: Params = Params(), level_lo=0, level_n=None):
    """
    step(), then start a uniformly drawn level when the episode ends. Levels are drawn from
    bank indices [level_lo, level_lo + level_n), the whole bank by default.
    """
    s, reward, done, truncated, info = step(bank, s, action, params)
    level_n = bank["seed"].shape[0] if level_n is None else level_n
    fresh = reset(bank, jax.random.randint(key, (), level_lo, level_lo + level_n))
    s = jax.tree_util.tree_map(lambda a, b: jnp.where(done | truncated, a, b), fresh, s)
    return s, reward, done, truncated, info
