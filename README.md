# RL GridWorld - Q-Learning Implementation

A reinforcement learning environment with Q-Learning agent for navigating a grid with obstacles.

## Approach

This implementation uses **Q-Learning**, a model-free reinforcement learning algorithm, to train an agent to navigate a grid world environment. Here's how it works:

### **Algorithm: Q-Learning**
- **Q-Table**: Stores expected future rewards for each (state, action) pair
- **Epsilon-Greedy Strategy**: Balances exploration (random actions) vs exploitation (learned actions)
- **Temporal Difference Learning**: Updates Q-values based on immediate rewards and future estimates

### **Environment Design**
- **GridWorld**: 10x10 grid with static and dynamic obstacles
- **Reward Structure**: +100 (goal), -1 (step), -10 (collision)
- **State Representation**: Grid coordinates (row, col)
- **Actions**: 4-directional movement (up, down, left, right)

### **Training Process**
1. **Exploration Phase**: Agent starts with (defined in the config)% random actions, gradually decreasing to 5%
2. **Learning**: Q-values updated using: `Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
3. **Evaluation**: Trained agent tested with minimal exploration (5%)
4. **Comparison**: Performance compared against random baseline

### **Features**
- **Modular Design**: Separate components for environment, agents, rewards, and metrics
- **Hyperparameter Optimization**: Automated grid search for optimal settings
- **Logging**: Training metrics, evaluation results, and visualizations
- **Path Visualization**: Visual representation of agent's navigation attempts

## Quick Start

### 1. Create Conda Environment
```bash
# Create a new conda environment with Python 3.9+
conda create -n rl_env python=3.9

# Activate the environment
conda activate rl_env
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Plotting and visualization
- `pandas>=1.3.0` - Data analysis and CSV handling

### 3. Run Training
```bash
python run.py
```

This will:
- Train the Q-Learning agent for 1500 episodes
- Evaluate the trained agent (10 episodes)
- Compare against a random baseline
- Generate visualizations and save to `results/YYYYMMDD_HHMMSS/`
- Create path visualizations for each evaluation episode

## Configuration

**All settings are in `config.py`**

### Environment Settings

```python
# In config.py
GRID_SIZE = 10              # Size of the grid (10x10)
START_STATE = (0, 0)        # Agent starts top-left
GOAL_STATE = (9, 9)         # Goal at bottom-right

N_STATIC_OBSTACLES = 7      # Red obstacles (stay same)
N_RANDOM_OBSTACLES = 3      # Orange obstacles (change each episode)
```

### Reward Values

```python
REWARD_GOAL = 100.0         # +100 for reaching goal
REWARD_STEP = -1.0          # -1 per step taken
REWARD_COLLISION = -10.0    # -10 for hitting wall/obstacle
```

### Training Hyperparameters

```python
N_EPISODES = 1500           # Number of training episodes
EPSILON_START = 1.0         # Start with 100% exploration
EPSILON_END = 0.05          # End with 5% exploration
EPSILON_DECAY = 0.999       # Decay rate

ALPHA = 0.1                 # Learning rate
GAMMA = 0.95                # Discount factor
```

### Evaluation Settings

```python
N_EVAL_EPISODES = 10         # Number of evaluation episodes
EVAL_EPSILON = 0.05         # 5% exploration during evaluation
MAX_EVAL_STEPS = 100        # Max steps before timeout
```

### Grid Search for Optimal Hyperparameters

Run automated hyperparameter optimization:

```bash
python grid_search.py
```

This will test different combinations of:
- Learning rate (alpha)
- Discount factor (gamma)
- Epsilon start/end values
- Epsilon decay rate
- Evaluation epsilon
- Q-value tolerance

Results saved to `results/grid_search/grid_search_YYYYMMDD_HHMMSS.csv`

### Custom Training Scripts

To Create training script:

```python
from environment import Environment
from agents import QLearningAgent
from run import train, evaluate

# Create environment
env = Environment(
    size=12,                    # Custom grid size
    n_static_obstacles=5,       # Custom obstacles
    n_random_obstacles=2
)

# Train with custom parameters
agent, logger = train(
    n_episodes=2000,
    epsilon_start=0.9,
    epsilon_end=0.02,
    alpha=0.2,
    gamma=0.98,
    env=env
)

# Evaluate
results = evaluate(agent, logger, env, n_episodes=20)
```

## Output Files

After running `python run.py`,the `results/` folder:

```
results/20251014_HHMMSS/
├── training_metrics.csv          # Episode-by-episode training data
├── evaluation_metrics.csv        # Evaluation results
├── eval_path_episode_1.png       # Visual of agent's path (success/fail)
├── eval_path_episode_2.png
├── ...
└── Various metric plots (.png)   # Learning curves, success rates, etc.
```

## Visualization Legend

### Path Visualizations
- **Red squares** - Static obstacles (same every episode)
- **Orange squares** - Random obstacles (change each episode)
- **Green square (G)** - Goal
- **Blue line with arrows** - Agent's path
- **Green circle** - Start position
- **Green star** - Goal reached!
- **Red X** - Failed/timed out

## Project Structure

```
rl_tech_test/
├── config.py                 # Default values/seeting
├── run.py                    # Main training script
├── environment.py            # GridWorld environment
├── agents/
│   └── QLearningAgent.py     # Q-Learning implementation
├── mdp/
│   ├── rewards.py            # Reward calculation
│   ├── observations.py       # State representation
│   └── terminations.py       # Episode end conditions
├── metrics/
│   ├── metrics.py            # Data logging
│   └── visualization.py      # Plot generation
└── results/                  # Training outputs
```