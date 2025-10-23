"""Base configuration file - Environment and general settings"""

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Grid settings
GRID_SIZE = 10
MAX_STEPS_PER_EPISODE = 100

# Starting positions
START_STATE = (0, 0)
GOAL_STATE = (9, 9)

# Obstacles
N_STATIC_OBSTACLES = 7
N_RANDOM_OBSTACLES = 3

# ============================================================================
# REWARD VALUES
# ============================================================================

REWARD_GOAL = 100.0  # Reward for reaching the goal
REWARD_STEP = -1.0  # Penalty for each step taken
REWARD_COLLISION = -10.0  # Penalty for hitting wall/obstacle

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

N_EVAL_EPISODES = 10  # Number of evaluation episodes
EVAL_EPSILON = 0.05  # Small exploration during evaluation
MAX_EVAL_STEPS = 100  # Maximum steps before timeout

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

FIGURE_SIZE = (8, 8)
PATH_FIGURE_SIZE = (10, 10)
DPI = 150

# ============================================================================
# COMPUTED VALUES
# ============================================================================

N_STATES = GRID_SIZE * GRID_SIZE  # Total possible states in grid
N_ACTIONS = 4  # Up, Down, Left, Right
