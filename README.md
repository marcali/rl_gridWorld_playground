# RL GridWorld - Q-Learning Implementation

A reinforcement learning environment with Q-Learning agent for navigating a grid with obstacles.

## 🎯 Quick Start

### 1. Install Dependencies
```bash
conda activate rl_env
pip install -r requirements.txt
```

### 2. Run Training
```bash
python run.py
```

This will:
- Train the Q-Learning agent
- Evaluate the trained agent
- Generate visualizations and save to `results/`
- Create path visualizations for each evaluation episode

## ⚙️ Configuration

**All settings are in `config.py`** - Change values there to customize your experiment!

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
N_EPISODES = 1000           # Number of training episodes
EPSILON_START = 1.0         # Start with 100% exploration
EPSILON_END = 0.05          # End with 5% exploration
EPSILON_DECAY = 0.999       # Slow decay

ALPHA = 0.3                 # Learning rate
GAMMA = 0.95                # Discount factor
```

### Evaluation Settings

```python
N_EVAL_EPISODES = 1         # Number of evaluation episodes
EVAL_EPSILON = 0.05         # 5% exploration during evaluation
MAX_EVAL_STEPS = 200        # Max steps before timeout
```

## 📊 Output Files

After running `python run.py`, check the `results/` folder:

```
results/20251014_HHMMSS/
├── training_metrics.csv          # Episode-by-episode training data
├── evaluation_metrics.csv        # Evaluation results
├── eval_path_episode_1.png       # Visual of agent's path (success/fail)
├── eval_path_episode_2.png
├── ...
└── Various metric plots (.png)   # Learning curves, success rates, etc.
```

## 🎨 Visualization Legend

### Path Visualizations
- 🔴 **Red squares** - Static obstacles (same every episode)
- 🟠 **Orange squares** - Random obstacles (change each episode)
- 🟢 **Green square (G)** - Goal
- 🔵 **Blue line with arrows** - Agent's path
- 🟢 **Green circle** - Start position
- ⭐ **Green star** - Goal reached!
- ❌ **Red X** - Failed/timed out



## 📁 Project Structure

```
rl_tech_test/
├── config.py                 # ⚙️ ALL SETTINGS HERE
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

