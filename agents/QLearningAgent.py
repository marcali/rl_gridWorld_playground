import numpy as np
import pickle
from pathlib import Path
from config import base_config, experiement_config as exp_config
from .agent_protocol import AgentProtocol


class QLearningAgent(AgentProtocol):
    def __init__(self, n_states, n_actions, alpha=exp_config.ALPHA, gamma=exp_config.GAMMA):
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(
            low=-exp_config.Q_TABLE_INIT_RANGE,
            high=exp_config.Q_TABLE_INIT_RANGE,
            size=(n_states, n_actions),
        )
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        # Convergence tracking
        self.q_value_history = []  # Store Q-value statistics over time
        self.episode_q_stats = []  # Store per-episode Q-value statistics

    def act(self, state, epsilon=0.1, tolerance=exp_config.Q_VALUE_TOLERANCE):
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

    def log_q_value_stats(self, episode):
        """Log Q-value statistics for convergence tracking"""
        # Calculate Q-value statistics
        q_values = self.q_table.flatten()
        q_stats = {
            "episode": episode,
            "mean_q": np.mean(q_values),
            "std_q": np.std(q_values),
            "max_q": np.max(q_values),
            "min_q": np.min(q_values),
            "q_range": np.max(q_values) - np.min(q_values),
            "non_zero_states": np.count_nonzero(np.any(self.q_table != 0, axis=1)),
            "total_states": self.q_table.shape[0],
        }
        self.episode_q_stats.append(q_stats)
        return q_stats

    def get_state_value_map(self):
        """Get the value of each state (max Q-value for each state)"""
        return np.max(self.q_table, axis=1)

    def get_convergence_metrics(self):
        """Get convergence-related metrics"""
        if len(self.episode_q_stats) < 2:
            return None

        # Calculate Q-value change over time
        recent_episodes = min(100, len(self.episode_q_stats))
        recent_mean_q = [stats["mean_q"] for stats in self.episode_q_stats[-recent_episodes:]]

        if len(recent_mean_q) < 2:
            return None

        # Calculate stability (how much Q-values are changing)
        q_std_change = np.std(np.diff(recent_mean_q))

        # Calculate convergence rate (how quickly Q-values are stabilizing)
        convergence_rate = 1.0 / (1.0 + q_std_change) if q_std_change > 0 else 1.0

        return {
            "q_value_stability": q_std_change,
            "convergence_rate": convergence_rate,
            "current_mean_q": recent_mean_q[-1],
            "q_value_trend": (
                np.polyfit(range(len(recent_mean_q)), recent_mean_q, 1)[0]
                if len(recent_mean_q) > 1
                else 0
            ),
        }

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
