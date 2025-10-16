"""Configuration file - All constants and hyperparameters in one place"""

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
# Q-LEARNING HYPERPARAMETERS
# ============================================================================

# Training parameters
N_EPISODES = 2000
EPSILON_START = 1.0  # Initial exploration rate (100% random)
EPSILON_END = 0.05  # Final exploration rate (5% random)
EPSILON_DECAY = 0.999  # Decay rate per episode

# Learning parameters
ALPHA = 0.15  # Learning rate (how much to update Q-values)
GAMMA = 0.95  # Discount factor (how much to value future rewards)

# Q-value tie-breaking tolerance
Q_VALUE_TOLERANCE = 0.01  # Tolerance for considering Q-values as "close"

# Q-table initialization
Q_TABLE_INIT_RANGE = 0.01  # Range for random Q-table initialization

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
# GRID SEARCH PARAMETERS
# ============================================================================

# Lists of values to search over
GRID_SEARCH_ALPHA = [0.05, 0.1, 0.2]
GRID_SEARCH_GAMMA = [0.95, 0.99]

GRID_SEARCH_EPSILON_DECAY = [0.995, 0.998, 0.999]  # Decay rates
GRID_SEARCH_EVAL_EPSILON = [0.05, 0.1, 0.1]
GRID_SEARCH_TOLERANCE = [0.001, 0.005, 0.01]  # Q-value tie-breaking tolerance

GRID_SEARCH_EPSILON_START = [0.8, 0.9, 1.0]  # Initial exploration rates
GRID_SEARCH_EPSILON_END = [0.01, 0.05, 0.1]  # Final exploration rates

# Grid search settings
GRID_SEARCH_N_EPISODES = 1000  # Episodes per configuration
GRID_SEARCH_N_EVAL = 20  # Evaluation episodes per configuration (increased for reliability)

# ============================================================================
# COMPUTED VALUES
# ============================================================================

N_STATES = GRID_SIZE * GRID_SIZE  # Total possible states in grid
N_ACTIONS = 4  # Up, Down, Left, Right
