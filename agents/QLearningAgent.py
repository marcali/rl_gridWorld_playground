import numpy as np
import pickle
from pathlib import Path
from config import base_config, qlearning_config
from .agent_protocol import AgentProtocol
from .convergence_tracking import ConvergenceTrackingMixin


class QLearningAgent(AgentProtocol, ConvergenceTrackingMixin):
    def __init__(
        self, n_states, n_actions, alpha=qlearning_config.ALPHA, gamma=qlearning_config.GAMMA
    ):
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(
            low=-qlearning_config.Q_TABLE_INIT_RANGE,
            high=qlearning_config.Q_TABLE_INIT_RANGE,
            size=(n_states, n_actions),
        )
        self.n_actions = n_actions
        self.state_size = n_states  # Required for mixin
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        # Counts for UCB exploration
        self.state_action_counts = np.zeros((n_states, n_actions), dtype=np.int64)
        self.state_counts = np.zeros(n_states, dtype=np.int64)

        # Initialize convergence tracking mixin
        self.__init_convergence_tracking__()

    def act(self, state, epsilon=0.1, tolerance=qlearning_config.Q_VALUE_TOLERANCE):
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

    def act_ucb(self, state, c: float = 1.0, tolerance=qlearning_config.Q_VALUE_TOLERANCE):
        """Select action using UCB1 over Q-values.

        UCB score: Q(s,a) + c * sqrt( ln(N(s) + 1) / (N(s,a) + 1) )
        """
        q_values = self.q_table[state]

        # Ensure at least 1 visit in denominators
        n_sa = self.state_action_counts[state] + 1
        n_s = self.state_counts[state] + 1

        bonus = c * np.sqrt(np.log(n_s) / n_sa)
        ucb_scores = q_values + bonus

        max_score = ucb_scores.max()
        tied_actions = np.flatnonzero(ucb_scores >= max_score - tolerance)
        return int(np.random.choice(tied_actions))

    def learn(self, state, action, reward, next_state, done, goal_state=None):
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

    def learn_ucb(self, state, action, reward, next_state, done, goal_state=None):
        """Update Q-values and visitation counts for UCB exploration."""
        # Increment visitation counts for UCB
        self.state_counts[state] += 1
        self.state_action_counts[state, action] += 1

        # Standard Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for a specific state"""
        return self.q_table[state, :]

    def save(self, path: str) -> None:
        """
        Save agent parameters to file

        Args:
            path: File path where to save the agent parameters
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        agent_data = {
            "q_table": self.q_table,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_value_history": self.q_value_history,
            "episode_q_stats": self.episode_q_stats,
            "state_action_counts": self.state_action_counts,
            "state_counts": self.state_counts,
        }

        # Save to file
        with open(path, "wb") as f:
            pickle.dump(agent_data, f)

    def load(self, path: str) -> None:
        """
        Load agent parameters from file

        Args:
            path: File path from where to load the agent parameters
        """
        # Check if file exists
        if not Path(path).exists():
            raise FileNotFoundError(f"Agent file not found: {path}")

        # Load data from file
        with open(path, "rb") as f:
            agent_data = pickle.load(f)

        # Restore agent parameters
        self.q_table = agent_data["q_table"]
        self.n_actions = agent_data["n_actions"]
        self.alpha = agent_data["alpha"]
        self.gamma = agent_data["gamma"]
        self.q_value_history = agent_data.get("q_value_history", [])
        self.episode_q_stats = agent_data.get("episode_q_stats", [])
        self.state_action_counts = agent_data.get(
            "state_action_counts", np.zeros_like(self.q_table, dtype=np.int64)
        )
        self.state_counts = agent_data.get(
            "state_counts", np.zeros(self.q_table.shape[0], dtype=np.int64)
        )
