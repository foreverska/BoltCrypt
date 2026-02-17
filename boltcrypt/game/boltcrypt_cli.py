import sys
import argparse
from boltcrypt.envs import BoltCrypt
from boltcrypt.wrapper import NaturalLanguage

# Fallback dimensions if not explicitly imported
MIN_ROOM_DIM = 4
MAX_ROOM_DIM = 10


def run_cli(config):
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

def main():
    parser = argparse.ArgumentParser(description="Play BoltCrypt CLI with custom DungeonGenerator settings.")

    # Core generation parameters
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the dungeon generation.")
    parser.add_argument("--min_dist", type=int, default=5, help="Minimum distance to the exit.")
    parser.add_argument("--max_dist", type=int, default=None, help="Maximum distance to the exit.")
    parser.add_argument("--mean_rooms", type=int, default=15, help="Average number of rooms to generate.")
    parser.add_argument("--std_rooms", type=int, default=2, help="Standard deviation for the room count.")
    parser.add_argument("--connectivity", type=float, default=0.3, help="Probability of creating loops in the map.")

    # Puzzle parameters
    parser.add_argument("--puzzle_density", type=float, default=0.3, help="Probability of a room containing a puzzle.")
    parser.add_argument("--key_puzzle_prob", type=float, default=0.3, help="Probability of the exit requiring a key.")
    parser.add_argument("--puzzle_required", action="store_true",
                        help="Forces the room before the exit to have a puzzle.")

    # Pass multiple arguments for allowed puzzles (e.g., --allowed_puzzles boulder stone)
    parser.add_argument("--allowed_puzzles", nargs="+", default=['boulder', 'mapped_plates', 'stone', 'warp_cycle'],
                        help="List of allowed puzzle types. Space separated.")

    # Room dimensions
    parser.add_argument("--min_room_dim", type=int, default=MIN_ROOM_DIM, help="Minimum width/height for a room.")
    parser.add_argument("--max_room_dim", type=int, default=MAX_ROOM_DIM, help="Maximum width/height for a room.")

    args = parser.parse_args()

    # Convert argparse Namespace directly to a dictionary to pass to generator_config
    config_dict = vars(args)

    run_cli(config_dict)

if __name__ == "__main__":
    main()