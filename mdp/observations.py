"""Observation processing for the MDP environment"""


class ObsTerm:
    """Defines how observations are computed from state,
    ***the task requires only to include current state, creating issue for obstacle avoidance during evaluation
    """

    def __init__(self):
        pass

    def __call__(self, state, env_size):
        """
        Convert environment state to observation

        Args:
            state: Current state of the environment (as [y, x])
            env_size: Environment size

        Returns:
            observation: Processed observation for the agent (flattened index)
        """
        y, x = state
        return y * env_size + x
