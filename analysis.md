# Analysis: RL GridWorld Q-Learning Implementation

## 1. What exploration strategy did you use and why?

The implementation uses Q-Learning, a model-free reinforcement learning algorithm that learns the optimal action-value function $Q(s,a)$ representing the expected future reward for taking action 'a' in state 's'. The Q-Learning update rule is Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)] where α is the learning rate, γ is the discount factor, r is the immediate reward, and s' is the next state after taking action 'a'. For exploration, I implemented an epsilon-greedy strategy with EPSILON_START = 1.0 (100% random actions initially), EPSILON_END = 0.05 (5% random actions at the end), and EPSILON_DECAY = 0.999 (decay rate per episode). The epsilon-greedy decision process works by generating a random number between 0 and 1, and if it's less than epsilon, the agent chooses a random action for exploration, otherwise it chooses the greedy action with the highest Q-value for exploitation.

The epsilon decay is used for the exploration-exploitation balance: during early training when the agent knows nothing, it needs high exploration to discover good actions, but during late training when the agent has learned, it needs high exploitation to use its learned knowledge. Without decay, the agent would continue taking random actions forever and not optimize. In other words, this means the agent starts with 100% exploration and 0% exploitation during early training, gradualy shifting the balance according to the decay rate, and finally reaches 5% exploration and 95% exploitation for final performance.

For GridWorld specifically, high initial exploration helps the agent learn to avoid walls and obstacles, random actions help discover paths to the goal location, and the final 5% exploration prevents overfitting to specific obstacle patterns. Due to the time constrains I have not considered other algorithms or strategies, rather focusing on easy and fast to implement hyperparameter grid search.This is a simple algorithm that is not designed to deal with random pop-up obstacles well and it can get stuck, for this reason the epsilon used for evaluation is 5% so that the agent can still take random actions if there is a new obstacle in the way.

---

## 2. How did you choose your hyperparameters (learning rate, epsilon, etc.)?

I chose the hyperparameters through a combination of empirical testing and automated grid search optimization. For the learning rate (α = 0.1), I found that values too high (>0.3) cause Q-values to oscillate with unstable learning, while values too low (<0.05) result in slow convergence requiring many episodes, so 0.1 provides a good balance between stability and learning speed. The discount factor (γ = 0.95) was chosen because it's high enough for the agent to consider long-term consequences. I tested the range 0.9-0.99 and found 0.95 performed best empirically. For the epsilon schedule, I used EPSILON_START = 1.0 (100% initial exploration), EPSILON_END = 0.05 (5% final exploration), and EPSILON_DECAY = 0.999 (decay rate), with these values optimized through grid search.

The environment parameters were given by the test, GRID_SIZE = 10, N_STATIC_OBSTACLES = 7, and N_RANDOM_OBSTACLES = 3. I used automated grid search to refine the hyperparameters, testing ranges of learning rates [0.05, 0.1, 0.2], discount factors [0.90, 0.95, 0.99], epsilon start values [0.8, 0.9, 1.0], and epsilon end values [0.01, 0.05, 0.1]. The selection criteria prioritized success rate (can the agent reach the goal) as the primary metric, followed by average steps (efficiency) as secondary, and average reward (overall performance) as tertiary.

Also I think maybe modifying steps and rewards engineering might improve performance but they were given in the task.

---

## 3. What was the biggest challenge you faced?

The biggest challenge was the agent getting stuck in loops, which is still not fully resolved and continues to happen during evaluation. This occurs when the agent repeatedly takes the same sequence of actions that don't lead to the goal, often due to the random obstacles changing the optimal path or the agent getting trapped in local optima. I tackled this issue by increasing the evaluation epsilon to 5% so the agent can still take random actions when encountering new obstacles or getting stuck, and by implementing a simple tie-breaking mechanism in the Q-value selection to add some randomness when multiple actions have similar Q-values. However, the fundamental issue remains because the agent has no memory of past moves within an episode, which creates problems when it needs to avoid previously visited states or remember that a certain path led to a dead end. This lack of memory means the agent can repeatedly explore the same failed paths without learning from its immediate history, leading to inefficient navigation and loop formation that significantly impacts performance.


## 4. If you had more time, what would you improve?

If I had more time, I would focus on several key improvements to address the current limitations and enhance the system's robustness. First, I would try other algorithms beyond Q-Learning, such as Deep Q-Networks (DQN) for larger grids, Double DQN to reduce overestimation bias, and policy gradient methods like Actor-Critic for more complex environments. I would also explore better exploration strategies like UCB-based exploration, curiosity-driven exploration with intrinsic motivation, and adaptive exploration rates based on learning progress to better handle the loop problem. Most importantly, I would implement memory mechanisms to allow the agent to remember past moves within episodes, potentially using recurrent neural networks or state-action history tracking to avoid repeatedly exploring failed paths. For more robust evaluation, I would implement statistical significance testing to validate performance differences, cross-validation across different environment configurations to test generalization, benchmark comparisons with other algorithms like SARSA and Actor-Critic, and robustness testing with noise and perturbations. I would also reach comprehensive unit testing for all components including the environment, agents, rewards, and metrics to ensure code reliability and catch bugs early. Additionally, I would enhance the visualization system with real-time training progress and better debugging tools to understand the agent's decision-making process.

---

## Mathematical Formulas and Equations

### **Q-Learning Algorithm**

**Q-Value Update Rule:**

$Q(s,a) = Q(s,a) + \alpha[r + \gamma \max Q(s',a') - Q(s,a)]$

- $Q(s,a)$: Current Q-value for state $s$ and action $a$
- $\alpha$: Learning rate (how much to update)
- $r$: Immediate reward received
- $\gamma$: Discount factor (importance of future rewards)
- $s'$: Next state after taking action $a$
- $\max Q(s',a')$: Maximum Q-value in next state (best future action)

**Temporal Difference Target:**

$TD_{target} = r + \gamma \max Q(s',a') \times (1 - done)$

- $done$: Binary flag (1 if episode ends, 0 otherwise)
- When episode ends, future rewards are 0

**Temporal Difference Error:**

$TD_{error} = TD_{target} - Q(s,a)$


### **Exploration Strategy (Epsilon-Greedy)**

**Epsilon Decay:**

$\varepsilon(t+1) = \max(\varepsilon_{end}, \varepsilon(t) \times \varepsilon_{decay})$

- $\varepsilon(t)$: Epsilon value at episode $t$
- $\varepsilon_{end}$: Minimum epsilon value (5%)
- $\varepsilon_{decay}$: Decay rate per episode (0.999)

**Action Selection:**

$action = \begin{cases}
random\_action & \text{if } rand() < \varepsilon \\
\arg\max Q(s,a) & \text{otherwise}
\end{cases}$


### **Reward Structure**

**Step Penalty:**

$r_{step} = -1.0$


**Collision Penalty:**

$r_{collision} = -10.0$


**Goal Reward:**

$r_{goal} = +100.0$


**Total Episode Reward:**

$R_{total} = \sum(r_{step} + r_{collision} + r_{goal})$


### **State Representation**

**State Index Calculation:**

$state = row \times GRID\_SIZE + col$

- $row$: Grid row position (0 to $GRID\_SIZE-1$)
- $col$: Grid column position (0 to $GRID\_SIZE-1$)
- $GRID\_SIZE$: Size of grid (10×10 = 100 states)

**Action Encoding:**

$0 = UP (North)$
$1 = DOWN (South)$
$2 = LEFT (West)$
$3 = RIGHT (East)$


### **Convergence Metrics**

**Q-Value Statistics:**

$\text{mean}_Q = \frac{1}{N} \sum Q(s,a)$          # Average Q-value across all state-action pairs
$\text{std}_Q = \sqrt{\frac{1}{N} \sum (Q(s,a) - \text{mean}_Q)^2}$  # Standard deviation of Q-values
$\text{max}_Q = \max(Q(s,a))$              # Maximum Q-value
$\text{min}_Q = \min(Q(s,a))$              # Minimum Q-value
$Q_{range} = \text{max}_Q - \text{min}_Q$          # Range of Q-values


**State Value (V-function):**

$V(s) = \max Q(s,a)$                # Best action value for each state


**Convergence Rate:**

$\text{recent\_mean}_Q = [\text{mean}_Q^{episode_{t-99}}, ..., \text{mean}_Q^{episode_t}]$
$Q_{std\_change} = \text{std}(\text{diff}(\text{recent\_mean}_Q))$
$\text{convergence\_rate} = \frac{1.0}{1.0 + Q_{std\_change}}$

- Higher convergence rate (closer to 1.0) = more stable learning
- Lower convergence rate (closer to 0.0) = still learning rapidly

**Explored States:**

$\text{explored\_states} = \text{count}(Q(s,a) \neq 0)$  # Number of states with non-zero Q-values


### **Termination Conditions**

**Episode Termination:**

$done = (state == goal\_state) \lor (steps \geq MAX\_STEPS) \lor (collision\_with\_obstacle)$


**Environment Reset:**

$\text{new\_random\_obstacles} = \text{generate\_random\_positions}(N\_RANDOM\_OBSTACLES)$
$grid = \text{add\_obstacles}(\text{static\_obstacles} + \text{new\_random\_obstacles})$
$agent\_position = START\_STATE$


### **Observation Space**

**State Observation:**

$observation = current\_state\_index$  # Integer from 0 to 99


**Grid Observation:**

$grid[i,j] = \begin{cases}
0: & \text{free\_space} \\
1: & \text{static\_obstacle} \\
2: & \text{random\_obstacle}
\end{cases}$


### **Performance Metrics**

**Success Rate:**

$\text{success\_rate} = \frac{\text{successful\_episodes}}{\text{total\_episodes}} \times 100\%$


**Average Reward:**

$\text{avg\_reward} = \frac{1}{N} \sum \text{episode\_reward}$


**Average Steps:**

$\text{avg\_steps} = \frac{1}{N} \sum \text{episode\_steps}$


**Moving Average (Training):**

$\text{moving\_avg\_reward} = \frac{1}{100} \sum \text{episode\_reward\_last\_100\_episodes}$


### **Tie-Breaking Mechanism**

**Action Selection with Tolerance:**

$\max_Q = \max(Q(s,:))$
$\text{tied\_actions} = \{a: Q(s,a) \geq \max_Q - \text{tolerance}\}$
$\text{selected\_action} = \text{random\_choice}(\text{tied\_actions})$


### **Grid Search Optimization**

**Parameter Combinations:**

$\text{total\_combinations} = |ALPHA| \times |GAMMA| \times |EPSILON\_START| \times |EPSILON\_END| \times |EPSILON\_DECAY| \times |EVAL\_EPSILON| \times |TOLERANCE|$


**Hyperparameter Ranges:**

$ALPHA \in [0.05, 0.1, 0.2]$
$GAMMA \in [0.90, 0.95, 0.99]$
$EPSILON\_START \in [0.8, 0.9, 1.0]$
$EPSILON\_END \in [0.01, 0.05, 0.1]$
$EPSILON\_DECAY \in [0.995, 0.998, 0.999]$


---
