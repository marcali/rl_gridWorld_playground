

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
