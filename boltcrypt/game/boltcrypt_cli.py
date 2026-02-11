import sys
from boltcrypt.environ import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage


def run_cli():
    # Full Feature Config
    config = {
        'min_dist': 5,
        'mean_rooms': 15,
        'puzzle_density': 0.3,
        'key_puzzle_prob': 0.4,
        'allowed_puzzles': ["mapped_plates", "stone", "warp_cycle"]
    }

    env = BoltCrypt(generator_config=config)
    env = NaturalLanguage(env)  # Apply wrapper

    # Start
    obs, _ = env.reset()
    print(obs, end="")  # Wrapper adds its own prompts

    while True:
        try:
            # Get input
            user_input = input().strip()

            if not user_input:
                continue

            # Step
            obs, reward, done, trunc, info = env.step(user_input)

            # Render
            print(obs, end="")

            if done:
                # If it was an EXIT command, the obs handles the message
                # If it was a win, the obs handles the message
                break

        except KeyboardInterrupt:
            print("\nForce Quit.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break


if __name__ == "__main__":
    run_cli()