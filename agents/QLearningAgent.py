import numpy as np
import config
from .base import AgentProtocol


class QLearningAgent(AgentProtocol):
    def __init__(self, n_states, n_actions, alpha=config.ALPHA, gamma=config.GAMMA):
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(
            low=-config.Q_TABLE_INIT_RANGE,
            high=config.Q_TABLE_INIT_RANGE,
            size=(n_states, n_actions),
        )
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def act(self, state, epsilon=0.1, tolerance=config.Q_VALUE_TOLERANCE):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-table
            q_values = self.q_table[state]

            # Find all actions with Q-values close to max (within tolerance)
            max_q = q_values.max()
            tied_actions = np.flatnonzero(q_values >= max_q - tolerance)

            # Pick randomly among tied actions
            return int(np.random.choice(tied_actions))

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values based on experience"""
        # select action with highest Q-value
        best_next_action = np.argmax(self.q_table[next_state])
        # td = r + gamma * max_a Q(s', a) * (1 - done)
        # done = 1 if episode is done, 0 otherwise to make td just reward when reached terminal state, otherwise do update for next state
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done)
        # td error = td_target - Q(s, a)
        td_error = td_target - self.q_table[state, action]
        # update q-value=Q(s, a) + alpha * td_error
        self.q_table[state, action] += self.alpha * td_error
