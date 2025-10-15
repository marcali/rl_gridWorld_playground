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

## 🛠️ Customization Examples

### Easy Problem (Faster Training)
```python
# In config.py
GRID_SIZE = 5
N_STATIC_OBSTACLES = 2
N_RANDOM_OBSTACLES = 1
N_EPISODES = 500
```

### Hard Problem (More Challenge)
```python
# In config.py
GRID_SIZE = 15
N_STATIC_OBSTACLES = 15
N_RANDOM_OBSTACLES = 5
N_EPISODES = 5000
```

### More Aggressive Learning
```python
# In config.py
ALPHA = 0.5             # Faster learning
EPSILON_DECAY = 0.995   # Faster decay
```

### More Conservative Learning
```python
# In config.py
ALPHA = 0.1             # Slower, more stable
EPSILON_DECAY = 0.9995  # Slower decay, more exploration
```

## 🔍 Hyperparameter Tuning

### Run Grid Search

Find optimal hyperparameters automatically:

```bash
python grid_search.py
```

This will:
- Test all combinations of alpha, gamma, and epsilon_decay
- Train and evaluate each configuration
- Save results to `results/grid_search/`
- Show ranked table of results
- Recommend best hyperparameters

**Configure search space in `config.py`:**
```python
GRID_SEARCH_ALPHA = [0.1, 0.2, 0.3, 0.5]
GRID_SEARCH_GAMMA = [0.90, 0.95, 0.99]
GRID_SEARCH_EPSILON_DECAY = [0.995, 0.998, 0.999]
```

### How Best Hyperparameters Are Selected:

1. **Primary:** Evaluation success rate (highest = best)
2. **Secondary:** Average steps (lowest = most efficient)
3. **Validation:** Consistent training and evaluation performance

**Example output:**
```
Rank  Alpha  Gamma  Eps.Decay | Train%  Eval%   Steps   Reward
1     0.30   0.95   0.999      | 72.0    75.0    24.3    85.7
2     0.20   0.95   0.999      | 68.5    70.0    26.1    81.3
3     0.30   0.90   0.998      | 65.0    65.0    28.5    76.4
```

## 📈 Analyzing Results

### View Learning Curves
The training generates plots automatically. Check the results folder for:
- Learning curve (reward over time)
- Success rate progression
- Reward component breakdown
- Steps per episode

### Understanding Performance

**Good signs:**
- Success rate >70% in last 100 episodes
- Training and evaluation success rates are similar
- Rewards increasing over time
- Steps decreasing over time

**Need more training:**
- Success rate still climbing at end
- Large gap between training and evaluation success rates
- High variance in rewards

## 🔧 Troubleshooting

### Agent gets stuck in loops
→ Increase `EVAL_EPSILON` to 0.1 or 0.15

### Agent not learning
→ Increase `N_EPISODES` to 3000+
→ Increase `EPSILON_DECAY` to 0.9995

### Training too slow
→ Decrease `N_EPISODES` to 500
→ Increase `EPSILON_DECAY` to 0.995

### Want different rewards
→ Change `REWARD_*` values in config.py

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

## 🎓 Key Features

✅ **Centralized configuration** - Change one value in `config.py` affects entire system
✅ **Automatic visualization** - Path tracking for each evaluation episode
✅ **Comprehensive metrics** - Reward breakdown, success rates, performance tracking
✅ **Tuple-based states** - Efficient, immutable state representation
✅ **Random obstacles** - Tests agent's ability to adapt
✅ **Consistent evaluation** - Uses same static obstacles as training

---

**To change ANY setting:** Edit `config.py` and re-run `python run.py`

