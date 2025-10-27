"""Trainer Configuration"""

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Episode Configuration
N_EPISODES = 1500  # Number of training episodes
EVAL_EPSILON = 0.05  # Exploration rate during evaluation
MAX_EVAL_STEPS = 100  # Maximum steps before timeout during evaluation

# Progress Reporting
PRINT_FREQUENCY = 100  # Print progress every N episodes
SAVE_FREQUENCY = 500  # Save model every N episodes (0 = never)

# Early Stopping
EARLY_STOPPING_PATIENCE = 200  # Stop if no improvement for N episodes
EARLY_STOPPING_MIN_DELTA = 0.01  # Minimum improvement to reset patience

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Evaluation Configuration
N_EVAL_EPISODES = 10  # Number of evaluation episodes
EVAL_EPSILON = 0.05  # Exploration rate during evaluation
MAX_EVAL_STEPS = 100  # Maximum steps per evaluation episode

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Visualization Settings
SAVE_PATHS = True  # Save path visualizations
SAVE_METRICS = True  # Save training metrics
SAVE_MODELS = True  # Save trained models

# Path Visualization
PATH_SAVE_FREQUENCY = 1  # Save path every N episodes (0 = never)
MAX_PATH_EPISODES = 10  # Maximum number of path episodes to save

# Metrics Visualization
METRICS_SAVE_FREQUENCY = 1  # Save metrics every N episodes
PLOT_FREQUENCY = 100  # Generate plots every N episodes
