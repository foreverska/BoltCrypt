"""
Counter-based randomness for in-episode dynamics (sailing stones, random warps).

Every random draw made while stepping is a pure function of
(level_seed, episode_step, purpose, index), so a trajectory depends only on the level and
the action sequence. Unlike a stateful RNG, this is straightforward to reimplement bit-for-bit
with uint32 arithmetic (e.g. in JAX), which lets alternative implementations be checked for exact
parity with this one. Changing anything in this file changes every stochastic trajectory.
"""
import itertools

MASK32 = 0xFFFFFFFF

# Purpose tags keep independent draws in the same step uncorrelated
NOISE_STONE = 1
NOISE_WARP = 2

# All orderings of the four move directions, in itertools (lexicographic) order
DIRECTION_PERMS = tuple(itertools.permutations(range(4)))


def mix32(x: int) -> int:
    """lowbias32 integer hash (C. Wellons). All arithmetic is modulo 2**32."""
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    x ^= x >> 16
    return x


def step_noise(level_seed: int, step: int, purpose: int, index: int = 0) -> int:
    """Uniform uint32 for one draw. level_seed is reduced modulo 2**32."""
    h = mix32(level_seed)
    h = mix32(h ^ (step & MASK32))
    h = mix32(h ^ purpose)
    return mix32(h ^ (index & MASK32))


def step_randint(n: int, level_seed: int, step: int, purpose: int, index: int = 0) -> int:
    """Integer in [0, n). Plain modulo (slightly biased) so other implementations match exactly."""
    return step_noise(level_seed, step, purpose, index) % n
