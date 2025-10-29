"""Main training script - Entry point that orchestrates training and evaluation"""

from enviroment import Environment
from agents import QLearningAgent, DQNAgent
from training import Trainer, Evaluator
from metrics.visualization import create_all_visualizations
from mdp import GoalReachedReward, StepPenaltyReward, CollisionPenaltyReward, RandomObstacleEvent
from config import base_config, dqn_config, qlearning_config, trainer_config


if __name__ == "__main__":
    # Create environment with specific rewards and random obstacles
    env = Environment(
        rewards=[
            GoalReachedReward(base_config.REWARD_GOAL),
            StepPenaltyReward(base_config.REWARD_STEP),
            CollisionPenaltyReward(base_config.REWARD_COLLISION),
        ],
        event_terms=[RandomObstacleEvent(n_obstacles=base_config.N_RANDOM_OBSTACLES)],
    )

    # Create q learning agent with default parameters
    agent = QLearningAgent(
        n_states=base_config.N_STATES,
        n_actions=base_config.N_ACTIONS,
        alpha=qlearning_config.ALPHA,
        gamma=qlearning_config.GAMMA,
    )

    # Create dqn agent with HER-compatible state size
    # When HER is enabled, state space becomes grid_size^4 (goal-conditioned)
    # her_state_size = (
    #     base_config.N_STATES * base_config.N_STATES if True else base_config.N_STATES
    # )  # HER enabled

    # agent = DQNAgent(
    #     state_size=her_state_size,
    #     action_size=base_config.N_ACTIONS,
    #     learning_rate=dqn_config.LEARNING_RATE,
    #     gamma=dqn_config.GAMMA,
    #     epsilon=dqn_config.EPSILON_START,
    #     epsilon_min=dqn_config.EPSILON_END,
    #     epsilon_decay=dqn_config.EPSILON_DECAY,
    #     batch_size=dqn_config.BATCH_SIZE,
    #     memory_size=dqn_config.MEMORY_SIZE,
    #     target_update_freq=dqn_config.TARGET_UPDATE_FREQ,
    #     hidden_size=dqn_config.HIDDEN_SIZE,
    # )

    # Create trainer and evaluator
    trainer = Trainer()
    evaluator = Evaluator()

    # Train agent with HER enabled
    print("Training agent with HER...")
    # trained_agent, metrics_logger = trainer.train(
    #     agent=agent,
    #     env=env,
    #     n_episodes=trainer_config.N_EPISODES,
    #     epsilon_start=dqn_config.EPSILON_START,
    #     epsilon_end=dqn_config.EPSILON_END,
    #     epsilon_decay=dqn_config.EPSILON_DECAY,
    #     silent=False,
    #     # HER parameters
    #     use_her=True,
    #     her_k=4,
    #     her_strategy="future",
    # )
    print("Training q learning agent...")
    trained_agent, metrics_logger = trainer.train(
        agent=agent,
        env=env,
        n_episodes=trainer_config.N_EPISODES,
        epsilon_start=qlearning_config.EPSILON_START,
        epsilon_end=qlearning_config.EPSILON_END,
        epsilon_decay=qlearning_config.EPSILON_DECAY,
    )

    # Evaluate agent with random baseline comparison
    print("Evaluating agents...")
    # metrics_logger, trained_results, random_results = evaluator.evaluate_with_random_baseline(
    #     agent=trained_agent,
    #     logger=metrics_logger,
    #     env=env,
    #     n_episodes=trainer_config.N_EVAL_EPISODES,
    #     eval_epsilon=dqn_config.EVAL_EPSILON,
    #     tolerance=qlearning_config.Q_VALUE_TOLERANCE,
    # )

    metrics_logger, trained_results, random_results = evaluator.evaluate_with_random_baseline(
        agent=trained_agent,
        logger=metrics_logger,
        env=env,
        n_episodes=trainer_config.N_EVAL_EPISODES,
        eval_epsilon=qlearning_config.EVAL_EPSILON,
        tolerance=qlearning_config.Q_VALUE_TOLERANCE,
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_all_visualizations(
        metrics_logger.get_experiment_dir(), trained_results, random_results, trained_agent, env
    )
