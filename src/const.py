# Constants used in the Angry Birds MDP environment.

# Frames per second for the game loop.
FPS = 12

# Colors for different grid tiles.
COLORS = {
    'T': (135, 206, 235),  # Tile ground
    'P': (135, 206, 235),  # Pigs
    'Q': (135, 206, 235),  # Queen
    'G': (135, 206, 235),  # Goal
    'R': (135, 206, 235),  # Rock
}

# Reward values for different tile types.
GOOD_PIG_REWARD = 250  # Reward for capturing a pig.
GOAL_REWARD = 400      # Reward for reaching the goal tile (egg).
QUEEN_REWARD = -400    # Penalty for encountering a queen pig.
DEFAULT_REWARD = -1    # Default reward for other tiles.

# Number of specific objects in the grid.
PIGS = 8       # Number of pigs in the environment.
QUEENS = 2     # Number of queen pigs in the environment.
ROCKS = 8      # Number of rocks (obstacles) in the environment.

# Definitions of movement actions.
actions = [
    (-1, 0),  # Up: Move one tile upward.
    (1, 0),   # Down: Move one tile downward.
    (0, -1),  # Left: Move one tile to the left.
    (0, 1)    # Right: Move one tile to the right.
]

# Mapping from movement vectors to action indices.
action_codes = {
    (-1, 0): 0,  # Up
    (1, 0): 1,   # Down
    (0, -1): 2,  # Left
    (0, 1): 3    # Right
}
