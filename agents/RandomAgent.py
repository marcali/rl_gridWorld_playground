import numpy as np
import pickle
from pathlib import Path
from .agent_protocol import AgentProtocol


class RandomAgent(AgentProtocol):
    """Agent that takes completely random actions - used as baseline"""

    def __init__(self, n_states, n_actions):
        """
        Initialize random agent

        Args:
            n_states: Number of states (not used, but kept for consistency)
            n_actions: Number of possible actions
        """
        self.n_actions = n_actions

    def act(self, state, epsilon=None, tolerance=None):
        """
        Select a random action

        Args:
            state: Current state (ignored)
            epsilon: Exploration rate (ignored)
            tolerance: Q-value tolerance (ignored)

        Returns:
            Random action (integer from 0 to n_actions-1)
        """
        return np.random.randint(self.n_actions)

    def learn(self, state, action, reward, next_state, done, goal_state=None):
        """
        Random agent doesn't learn

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            goal_state: Goal state
        """
        pass

    def save(self, path: str) -> None:
        """
        Save agent parameters to file

        Args:
            path: File path where to save the agent parameters
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save (RandomAgent only has n_actions)
        agent_data = {"n_actions": self.n_actions}

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
        self.n_actions = agent_data["n_actions"]
