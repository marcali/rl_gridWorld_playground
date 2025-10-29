"""Training module for RL agents"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from agents import QLearningAgent, AgentProtocol
from metrics import MetricsLogger
from config import base_config, qlearning_config, trainer_config
from experience.her import HindsightExperienceReplay
from experience.goal_extraction import GridWorldGoalExtractor


class Trainer:
    """Handles training of RL agents"""

    def __init__(self, experiment_name=None):
        """
        Initialize trainer

        Args:
            experiment_name: Name for this experiment
        """
        self.experiment_name = experiment_name

    def train(
        self,
        agent: AgentProtocol,
        env,
        n_episodes,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        silent=False,
        use_her=False,
        her_k=4,
        her_strategy="future",
    ):
        """
        Train the agent with metrics logging

        Args:
            agent: Pre-initialized agent to train
            env: Environment to train in
            n_episodes: Number of training episodes
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            silent: If True, suppress progress output for grid search
            use_her: If True, wrap agent's replay buffer with HER
            her_k: Number of hindsight goals per episode
            her_strategy: HER strategy ('future', 'final', 'random')

        Returns:
            tuple: (agent, logger, success_rate) if silent, else (agent, logger)
        """
        # Validate required parameters
        if agent is None:
            raise ValueError("Agent must be provided. Please initialize an agent first.")

        if env is None:
            raise ValueError(
                "Environment must be provided. Please initialize an environment first."
            )

        # Only create logger if not in silent mode
        logger = (
            MetricsLogger(experiment_name=self.experiment_name, agent_type=agent.__class__.__name__)
            if not silent
            else None
        )

        # Setup HER wrapper if requested
        original_memory = None
        if use_her:
            # Create goal extractor for GridWorld
            goal_extractor = GridWorldGoalExtractor(env.size)

            # Wrap the agent's replay buffer with HER
            original_memory = agent.memory
            agent.memory = HindsightExperienceReplay(
                replay_buffer=original_memory,
                k=her_k,
                strategy=her_strategy,
                goal_extraction_func=goal_extractor.extract_achieved_goal,
                desired_goal_extraction_func=lambda state, goal_state: goal_extractor.extract_desired_goal(
                    state, goal_state
                ),
                goal_substitution_func=goal_extractor.substitute_goal,
            )

        epsilon = epsilon_start
        # automatically deletes oldest item when maxlen is reached, only keep last 100 items
        reward_history = deque(maxlen=100)
        success_history = deque(maxlen=100)

        # Training loop
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            # Track reward components
            goal_rewards = 0
            step_penalties = 0
            collision_penalties = 0

            while not done:
                # Select action
                action = agent.act(state, epsilon=epsilon)

                # Take step
                next_state, reward, done = env.step(action)

                # Learn from experience - unified approach
                # Agent handles both regular and HER cases internally
                agent.learn(state, action, reward, next_state, done, env.goal_state)

                # Track reward components
                if reward == base_config.REWARD_GOAL:  # Goal reached
                    goal_rewards += reward
                elif reward == base_config.REWARD_COLLISION:  # Collision
                    collision_penalties += reward
                elif reward == base_config.REWARD_STEP:  # Step
                    step_penalties += reward

                # Update tracking
                episode_reward += reward
                steps += 1
                state = next_state

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Check if agent reached goal
            success = env.current_state == env.goal_state

            # Update histories
            reward_history.append(episode_reward)
            success_history.append(success)
            avg_reward = np.mean(reward_history)
            success_rate = np.mean(success_history) * 100

            # Log Q-value statistics and convergence metrics (if available)
            if hasattr(agent, "log_q_value_stats"):
                q_stats = agent.log_q_value_stats(episode)
            else:
                q_stats = None

            if hasattr(agent, "get_convergence_metrics"):
                convergence_metrics = agent.get_convergence_metrics()
            else:
                convergence_metrics = None

            convergence_rate = (
                convergence_metrics["convergence_rate"] if convergence_metrics else 0.0
            )

            # Log metrics
            if logger is not None:
                logger.log_training_episode(
                    episode=episode,
                    total_reward=episode_reward,
                    steps=steps,
                    avg_reward_last_100=avg_reward,
                    success=success,
                    goal_rewards=goal_rewards,
                    step_penalties=step_penalties,
                    collision_penalties=collision_penalties,
                    q_stats=q_stats,
                    convergence_rate=convergence_rate,
                )

            # Print progress (only if not silent)
            if not silent and episode % 100 == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Average reward(100 episodes): {avg_reward:7.2f} | "
                    f"Steps: {steps:4d} | "
                    f"Success: {success_rate:5.1f}% | "
                )
                # Show reward breakdown
                if episode > 0:
                    print(
                        f"              | Goal: {goal_rewards:+6.0f} | Step: {step_penalties:+6.0f} | Collision: {collision_penalties:+6.0f}"
                    )
                # Show convergence metrics (if available)
                if convergence_metrics and q_stats:
                    print(
                        f"              | Mean Q: {q_stats['mean_q']:6.2f} | Convergence: {convergence_rate:5.3f} | "
                        f"Explored States: {q_stats['non_zero_states']:3d}/{q_stats['total_states']}"
                    )

        # Restore original memory if HER was used
        if use_her and original_memory is not None:
            agent.memory = original_memory

        # Return values based on mode
        if silent:
            return agent, logger, success_rate
        else:
            return agent, logger

    def _train_dqn(self, agent) -> None:
        """Train DQN agent on a batch of experiences"""
        if len(agent.memory) < agent.batch_size:
            return

        # Sample experiences from replay buffer
        experiences = agent.memory.sample(agent.batch_size)

        # Convert experiences to tensors
        states = torch.stack([agent._state_to_tensor(exp.state).squeeze(0) for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack(
            [agent._state_to_tensor(exp.next_state).squeeze(0) for exp in experiences]
        )
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)

        # Current Q values
        current_q_values = agent.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            # max(1) returns the maximum value for each row (action dimension)
            # [0] returns the index of the maximum value
            # so next_q_values is the Q-value of the action with the highest Q-value for each state in the batch
            next_q_values = agent.target_network(next_states).max(1)[0]
            # target_q_values = rewards + (agent.gamma * next_q_values * ~dones)
            # ~dones is 1 if the episode is not done, 0 if it is done
            target_q_values = rewards + (agent.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # Track loss
        agent.loss_history.append(loss.item())
