"""DQN Agent Configuration"""

# ============================================================================
# DQN AGENT HYPERPARAMETERS
# ============================================================================

# Neural Network Architecture
HIDDEN_SIZE = 128  # Number of neurons in hidden layers
NUM_LAYERS = 3  # Number of hidden layers (excluding input/output)

# Learning Parameters
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
GAMMA = 0.95  # Discount factor for future rewards

# Experience Replay
MEMORY_SIZE = 10000  # Size of replay buffer
BATCH_SIZE = 32  # Batch size for training

# Target Network
TARGET_UPDATE_FREQ = 100  # Steps between target network updates

# Exploration Parameters
EPSILON_START = 1.0  # Initial exploration rate (100% random)
EPSILON_END = 0.01  # Final exploration rate (1% random)
EPSILON_DECAY = 0.999  # Epsilon decay rate per episode

# Training Parameters
MAX_EPISODES = 1500  # Maximum training episodes
EVAL_EPSILON = 0.05  # Exploration rate during evaluation

# ============================================================================
# GRID SEARCH PARAMETERS FOR DQN
# ============================================================================

# Learning rate search space
DQN_LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01]

# Batch size search space
DQN_BATCH_SIZES = [16, 32, 64, 128]

# Memory size search space
DQN_MEMORY_SIZES = [10000, 50000, 100000]

# Hidden size search space
DQN_HIDDEN_SIZES = [64, 128, 256, 512]

# Target update frequency search space
DQN_TARGET_UPDATE_FREQS = [50, 100, 200, 500]

# Epsilon decay search space
DQN_EPSILON_DECAYS = [0.99, 0.995, 0.999, 0.9995]
