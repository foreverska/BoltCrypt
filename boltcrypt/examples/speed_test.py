import time
from boltcrypt.envs import BoltCrypt


def benchmark_resets(env, resets=10000):
    """Measure how many resets can be performed per second."""
    start = time.time()

    for _ in range(resets):
        env.reset()

    elapsed = time.time() - start
    return resets / elapsed


def benchmark_steps(env, resets=10, duration=5.0):
    """Measure how many steps can be performed per second."""
    all_sps = []

    for _ in range(resets):
        count = 0
        start = time.time()
        end = start + duration

        env.reset()

        while time.time() < end:
            # Use a simple action (e.g., movement)
            _, _, done, truncated, _ = env.step(0)
            count += 1

            if done or truncated:
                env.reset()

        elapsed = time.time() - start
        all_sps.append(count / elapsed)


    return sum(all_sps) / len(all_sps)


if __name__ == "__main__":
    config = {
        'min_dist': 10,
        'mean_rooms': 30,
        'puzzle_density': 0.3,
        'key_puzzle_prob': 0.4,
    }

    env = BoltCrypt(generator_config=config)

    print("Running benchmarks...")
    print(f"Resets per second: {benchmark_resets(env):.2f}")
    print(f"Steps per second: {benchmark_steps(env):.2f}")
