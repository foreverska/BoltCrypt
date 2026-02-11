import pygame
import numpy as np
# Assuming the BoltCrypt class is in a file named boltcrypt.py
# If you are running this in a single file, just paste the BoltCrypt class above this.
from boltcrypt.envs.boltcrypt import BoltCrypt, Direction, TILE_STONE, TILE_WARP, TILE_KEY, TILE_BOULDER, TILE_SWITCH, TILE_EXIT, \
    TILE_DOOR, TILE_WALL, PuzzleType

# --- CONSTANTS & COLORS ---
COLORS = {
    0: (230, 215, 180),  # Floor (Dark Void)
    1: (40, 40, 40),  # Wall
    2: (139, 69, 19),  # Door (Wood)
    3: (0, 255, 100),  # Exit (Neon Green)
    4: (200, 50, 50),  # Switch (Red)
    5: (139, 69, 19),  # Boulder (Brown)
    6: (255, 215, 0),  # Key (Gold)
    7: (100, 100, 100),  # Stone (Grey)
    8: (148, 0, 211),  # Warp (Purple)
    'AGENT': (50, 150, 255),
    'BG': (10, 10, 10),
    'TEXT': (220, 220, 220),
    'LOCK': (255, 50, 50)  # Color for the lock icon
}

TILE_SIZE = 48
OFFSET_X, OFFSET_Y = 50, 50


def draw_lock(screen, rect, color):
    """Draws a padlock icon on top of a rect with a specific color."""
    # Shackle (Grey/Metal or matched to lock type)
    shackle_color = (200, 200, 200)
    pygame.draw.arc(screen, shackle_color,
                    (rect.centerx - 6, rect.top + 12, 12, 18),
                    0, 3.14, 2)
    # Body
    pygame.draw.rect(screen, color,
                     (rect.centerx - 8, rect.top + 18, 16, 16))
    # Keyhole (Black dot)
    pygame.draw.circle(screen, (0,0,0), (rect.centerx, rect.top + 26), 2)


def render_gym(screen, font, env, obs, total_reward, done, text_status):
    screen.fill(COLORS['BG'])
    grid = obs['grid']
    agent_pos = obs['agent_pos']
    global_pos = obs['global_pos']
    has_key = obs['inventory'] == 1

    rows, cols = grid.shape

    # 1. DRAW GRID
    # Use min() to handle rooms larger than max_room_dim
    render_h = min(env.curr_room.h, rows)
    render_w = min(env.curr_room.w, cols)

    for y in range(render_h):
        for x in range(render_w):
            rect = pygame.Rect(OFFSET_X + x * TILE_SIZE, OFFSET_Y + y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

            tile_id = grid[y, x]
            color = COLORS.get(tile_id, (255, 0, 255))

            # Base Floor
            pygame.draw.rect(screen, COLORS[0], rect)

            # Draw Tile Specifics
            if tile_id == TILE_WALL:
                pygame.draw.rect(screen, color, rect)

            if tile_id == TILE_DOOR:
                pygame.draw.rect(screen, color, rect)

                # --- LOCK LOGIC START ---
                # 1. Identify which door this is (North, South, East, or West)
                check_dir = None
                if y == 0:
                    check_dir = Direction.SOUTH
                elif y == env.curr_room.h - 1:
                    check_dir = Direction.NORTH
                elif x == 0:
                    check_dir = Direction.WEST
                elif x == env.curr_room.w - 1:
                    check_dir = Direction.EAST

                # 2. Check for Local Puzzle Lock (Boulder Puzzle not solved)
                # If the current room is a BOULDER room and not solved, exits are blocked.
                is_puzzle_locked = (env.curr_room.puzzle_type.name == "BOULDER" and not env.curr_room.is_solved)

                # 2b. Check for Boulder Plates Lock
                # For boulder_plates, only the door mapped to the active plate is unlocked
                is_boulder_plates_locked = False
                if env.curr_room.puzzle_type.name == "BOULDER_PLATES" and check_dir:
                    if env.curr_room.active_plate is None:
                        is_boulder_plates_locked = True
                    else:
                        unlocked_door = env.curr_room.plate_door_map.get(env.curr_room.active_plate)
                        if unlocked_door != check_dir:
                            is_boulder_plates_locked = True

                if is_puzzle_locked:
                    # Draw Red Lock (Requires solving the room)
                    draw_lock(screen, rect, (255, 50, 50))
                elif is_boulder_plates_locked:
                    # Draw Orange Lock (Requires moving boulder to correct plate)
                    draw_lock(screen, rect, (255, 140, 0))
                    # 3. Check for Neighbor Key Lock (Target room requires key)
                elif check_dir:
                    nx = env.gx + check_dir.value[0]
                    ny = env.gy + check_dir.value[1]
                    if (nx, ny) in env.generator.grid:
                        neighbor = env.generator.grid[(nx, ny)]
                        if neighbor.is_locked and not has_key:
                            # Draw Gold Lock (Requires Key)
                            draw_lock(screen, rect, (255, 215, 0))

            elif tile_id == TILE_BOULDER:
                pygame.draw.circle(screen, color, rect.center, TILE_SIZE // 2 - 4)

            elif tile_id == TILE_STONE:  # Sailing Stone
                pygame.draw.circle(screen, color, rect.center, TILE_SIZE // 2 - 6)
                pygame.draw.circle(screen, (0, 0, 0), rect.center, TILE_SIZE // 2 - 6, 2)  # outline

            elif tile_id == TILE_WARP:
                # Draw a little spiral or distinct marker
                pygame.draw.rect(screen, color, rect.inflate(-10, -10))
                pygame.draw.rect(screen, (255, 255, 255), rect.inflate(-10, -10), 1)

            elif tile_id == TILE_KEY:
                pygame.draw.circle(screen, color, rect.center, 8)
                pygame.draw.line(screen, color, rect.center, (rect.centerx + 5, rect.centery - 5), 2)

            elif tile_id == TILE_SWITCH:
                pygame.draw.rect(screen, color, rect.inflate(-15, -15))

            elif tile_id == TILE_EXIT:
                pygame.draw.rect(screen, color, rect)

    # 2. DRAW AGENT
    # Note: Agent Pos is (x, y).
    ax, ay = agent_pos
    agent_rect = pygame.Rect(OFFSET_X + ax * TILE_SIZE, OFFSET_Y + ay * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.circle(screen, COLORS['AGENT'], agent_rect.center, 14)
    pygame.draw.circle(screen, (255, 255, 255), agent_rect.center, 14, 2)

    # 3. HUD
    # Room Info
    curr_room = env.curr_room
    p_type = curr_room.puzzle_type.name if hasattr(curr_room, 'puzzle_type') else "none"

    p_type_map = {
        PuzzleType.NONE.name: "",
        PuzzleType.BOULDER.name: "Boulders",
        PuzzleType.BOULDER_PLATES.name: "Boulders",
        PuzzleType.STONE.name: "Wandering Stones",
        PuzzleType.WARP_CYCLE.name: "Warps",
        PuzzleType.WARP_RND.name: "Random Warps"
    }
    p_type = p_type_map.get(p_type, "")

    # Text Rendering
    lines = [
        f"Global Pos: {global_pos} | Local: {agent_pos}",
        f"Room Puzzle: {p_type}",
        f"Inventory: {'KEY FOUND' if has_key else 'None'}",
        f"Reward: {total_reward:.2f}",
        f"Status: {text_status}"
    ]

    # Legend
    legend_start_y = 600
    for i, line in enumerate(lines):
        c = COLORS['TEXT']
        if "KEY FOUND" in line: c = COLORS[6]
        surf = font.render(line, True, c)
        screen.blit(surf, (20, legend_start_y + i * 25))


def play_dungeon():
    # Config: Enable ALL puzzles for testing
    config = {
        'min_dist': 4,
        'mean_rooms': 12,
        'puzzle_density': 0.3,
        'key_puzzle_prob': 0.3,
        'min_room_dim': 5,
        'max_room_dim': 9,
        'allowed_puzzles': ["boulder", "mapped_plates", "stone", "warp_cycle"],
    }

    env = BoltCrypt(generator_config=config)
    obs, _ = env.reset()
    total_reward = 0

    pygame.init()
    screen_w, screen_h = 600, 750
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("BoltCrypt")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    running = True
    done = False
    status_msg = "Arrows to Move | R to Reset"

    while running:
        action = None

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0
                    done = False
                    status_msg = "Reset!"
                elif not done:
                    # Input Mapping
                    if event.key == pygame.K_UP:
                        action = 1  # North (Standard Gym usually)
                    elif event.key == pygame.K_DOWN:
                        action = 0  # South
                    elif event.key == pygame.K_RIGHT:
                        action = 2  # East
                    elif event.key == pygame.K_LEFT:
                        action = 3  # West

                    # Update Status based on input
                    if action is not None:
                        # Step Environment
                        obs, reward, done, trunc, info = env.step(action)
                        total_reward += reward

                        # Feedback
                        if reward > 0.5 and reward < 5.0:
                            status_msg = "Nice! (+Reward)"
                        elif reward <= -0.01:
                            status_msg = ""

                        if done: status_msg = "VICTORY!"

        # Render
        render_gym(screen, font, env, obs, total_reward, done, status_msg)
        pygame.display.flip()
        clock.tick(15)  # Cap FPS (Gym envs is instant, but visuals need time)

    pygame.quit()


if __name__ == "__main__":
    play_dungeon()