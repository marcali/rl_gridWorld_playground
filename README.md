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

## Curriculum Learning

The environment supports curriculum learning, allowing you to modify rewards over time based on steps or episodes. This is useful for gradually increasing difficulty or changing reward structure as the agent learns.

### Basic Curriculum Learning

The curriculum learning system allows you to modify rewards over time based on steps or episodes. You can create base rewards and define curriculum rules that trigger modifications at specific thresholds.

## Available Agents

The codebase includes several agent implementations:

### Q-Learning Agent
Traditional tabular Q-learning with epsilon-greedy exploration.

### Random Agent
Baseline agent that takes random actions for comparison.

#### Training Pattern:
The DQN agent follows the same pattern as other agents:
- **`learn()`**: Stores experiences in replay buffer, updates target network, decays epsilon
- **Training logic**: Handled by the trainer (detected by presence of `q_network` attribute)
- **`act()`**: Selects actions using epsilon-greedy policy

## Event System

The environment supports an event system for dynamic changes like obstacle generation:

### Basic Event Usage

```python
from enviroment import Environment
from mdp import RandomObstacleEvent, EventManager

# Create environment with custom events (only random obstacles)
env = Environment(
    event_terms=[
        RandomObstacleEvent(n_obstacles=3)     # Random obstacles at reset
    ]
)

# Or no events (static obstacles only, no random obstacles)
env = Environment()  # Only static obstacles, no events
```

### Event Types

- **`RandomObstacleEvent`**: Adds random obstacles that change each episode
- **`EventManager`**: Manages multiple event terms and applies them

**Note**: Static obstacles are handled separately and are not part of the event system.

### Custom Events

You can create custom event terms by inheriting from `EventTerm`:

```python
from mdp import EventTerm

class CustomEvent(EventTerm):
    def apply(self, grid, state, goal_state):
        # Your custom event logic
        return modified_grid
```

## Modular Agent Usage

The codebase now supports modular agent initialization, making it easy to create different scripts with different agents:

### Basic Usage

The codebase supports modular agent initialization, making it easy to create different scripts with different agents. You can create environments and agents, use trainers and evaluators, and save/load trained models. The modular reward system allows you to specify custom reward combinations when creating environments. The main scripts (`run.py` and `grid_search.py`) include random obstacle events for more challenging training.

### Example Scripts

- `example_qlearning.py` - Shows how to train a single Q-Learning agent
- `example_multi_agent.py` - Shows how to compare multiple agents with different hyperparameters
- `run.py` - Main training script with random obstacle events
- `grid_search.py` - Hyperparameter optimization with random obstacle events

## Configuration

**All settings are in the `config/` directory**

### Environment Settings

The environment configuration includes grid size, starting and goal positions, and obstacle settings.

### Reward Values

The reward system includes goal rewards, step penalties, and collision penalties with configurable values.

### Training Hyperparameters

Training parameters include episode count, epsilon settings for exploration, learning rate, and discount factor.

### Evaluation Settings

Evaluation parameters include episode count, exploration rate during evaluation, and maximum steps per episode.

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

You can create custom training scripts with different environments, agents, and hyperparameters. The system supports flexible configuration for grid size, obstacles, training episodes, and learning parameters.

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
├── agents/                 # Agent implementations
│   ├── agent_protocol.py   # Abstract base class for agents
│   ├── QLearningAgent.py   # Tabular Q-learning agent
│   ├── RandomAgent.py      # Random baseline agent
│   └── DQNAgent.py         # Deep Q-Network agent
├── enviroment/             # Environment implementations
│   ├── base_enviroment.py  # Abstract base environment
│   └── grid_world.py       # Grid world environment
├── mdp/                    # MDP components
│   ├── observations.py     # Observation functions
│   ├── rewards.py          # Reward functions
│   ├── terminations.py     # Termination conditions
│   ├── curriculum.py       # Curriculum learning
│   └── events.py           # Event system for dynamic changes
├── training/               # Training and evaluation
│   ├── trainer.py          # Training logic
│   └── evaluator.py        # Evaluation logic
├── metrics/                # Metrics and visualization
│   ├── metrics.py          # Performance metrics
│   └── visualization.py    # Plotting functions
├── utils/                  # Utility classes
│   ├── experience_replay.py # Standard experience replay buffer
│   └── hindsight_replay.py  # Hindsight experience replay buffer
├── config/                 # Configuration files
│   ├── base_config.py      # Base configuration
│   ├── dqn_config.py       # DQN agent configuration
│   ├── qlearning_config.py # Q-Learning agent configuration
│   └── trainer_config.py   # Training configuration
├── results/                # Training results and visualizations
├── run.py                  # Main training script
├── grid_search.py          # Hyperparameter grid search
└── requirements.txt        # Python dependencies
```

## Experience Module

The `experience` module provides comprehensive replay buffer implementations for off policy agents.

### Replay Buffers

#### Files:
- **`experience/replay_buffer.py`**: Comprehensive replay buffer implementations

#### Features:
- **ReplayBuffer**: Traditional FIFO replay buffer
- **Configurable capacity**: Adjustable buffer size
- **Statistics tracking**: Monitor buffer utilization and performance
- **Episode grouping**: Organize experiences by episodes