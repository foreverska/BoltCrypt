# Change Log #

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `lava_density`: lethal lava tiles in puzzle-free rooms. Stepping on one terminates the episode,
  and `info['died']` reports it. Tiles are kept only if doorways, the room centre and any key
  stay mutually reachable, and placement draws nothing when disabled. Lava is visible (`TILE_LAVA`
  = 10), drawn by `NaturalLanguage` (which no longer announces a victory on death), and supported
  by the JAX port under the parity test.
- `boltcrypt.jaxenv`: functional JAX port (`reset`/`step`/`observe`/`step_autoreset`) on a bank of
  pre-generated levels (`level_bank.build_bank`/`save_bank`/`load_bank`), with a step-by-step
  parity test against the Python env run in CI on the CPU backend. Optional extra `[jax]`.
- `trap_prob`: irreversible trap rooms. Entering one seals its doors for the rest of the episode, so
  the agent waits out the step limit. Traps are chosen after the rest of generation and are only
  kept if the start, key and exit stay connected around them. A seed gives the same layout with or
  without traps. `info['trapped']` reports a sprung trap, and `NaturalLanguage` describes it.
- Test suite (`pip install -e .[test]`, `pytest`) and a GitHub Actions workflow that runs it.

### Changed
- `boltcrypt.jaxenv.env.step` / `step_autoreset` return an info dict (`trapped`, `died`) in place
  of the bare trapped flag. `step_autoreset` can draw levels from a sub-range of the bank.
- Device banks keep the large per-tile arrays compact; warp candidate lists are sized to the bank.
- In-episode randomness (sailing stones, random warps) is now a pure function of
  (level seed, episode step) via `boltcrypt.envs.step_noise`, so other implementations can match
  trajectories exactly. Stochastic trajectories differ from 0.1.2 for the same seed.
- `reset()` without a seed draws the level from the env's own RNG, so a seeded first reset makes
  the whole sequence of levels reproducible. The level's seed is exposed as `env.level_seed`.

### Fixed
- The +1 puzzle bonus could be collected repeatedly by pushing a boulder off its plate and back on;
  it is now paid only the first time a room is solved.
- `connectivity` was inverted: 1.0 added no loops and small values added the most. Dungeons
  generated with any connectivity other than 0.0 differ from 0.1.2.
- `puzzle_required` could assign two puzzles to the room before the exit, mixing their state.
- `NaturalLanguage` crashed when describing a key-locked exit room, failed when stacked on another
  wrapper, reported every mapped-plate door as locked, and used unseeded flavour/hint text.
- `render()` now accepts the advertised `rgb_array` mode.

## [0.1.2] - 2026-02-16

### Added
- Ollama integration example
- Several quality of life improvements
- Bug fixes

## [0.1.1] - 2026-02-11

Initial release rerelease