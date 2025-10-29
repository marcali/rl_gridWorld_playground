"""Q-Learning Agent Configuration"""

# ============================================================================
# Q-LEARNING AGENT HYPERPARAMETERS
# ============================================================================

# Learning Parameters
ALPHA = 0.15  # Learning rate (how much to update Q-values)
GAMMA = 0.95  # Discount factor (how much to value future rewards)

# Exploration Parameters
EPSILON_START = 1.0  # Initial exploration rate (100% random)
EPSILON_END = 0.05  # Final exploration rate (5% random)
EPSILON_DECAY = 0.999  # Decay rate per episode

# Q-value Parameters
Q_VALUE_TOLERANCE = 0.01  # Tolerance for considering Q-values as "close"
Q_TABLE_INIT_RANGE = 0.01  # Range for random Q-table initialization

# Training Parameters
MAX_EPISODES = 1500  # Maximum training episodes
EVAL_EPSILON = 0.05  # Exploration rate during evaluation

# ============================================================================
# GRID SEARCH SETTINGS
# ============================================================================

# Grid search specific parameters
GRID_SEARCH_N_EPISODES = 1000  # Episodes per configuration
GRID_SEARCH_N_EVAL = 20  # Evaluation episodes per configuration

# Grid search parameter
GRID_SEARCH_ALPHA = [0.05, 0.1, 0.2]
GRID_SEARCH_GAMMA = [0.95, 0.99]
GRID_SEARCH_EPSILON_DECAY = [0.995, 0.998, 0.999]
GRID_SEARCH_EVAL_EPSILON = [0.05, 0.1, 0.1]
GRID_SEARCH_TOLERANCE = [0.001, 0.005, 0.01]
GRID_SEARCH_EPSILON_START = [0.8, 0.9, 1.0]
GRID_SEARCH_EPSILON_END = [0.01, 0.05, 0.1]
