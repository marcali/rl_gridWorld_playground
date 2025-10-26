"""Main training script - Entry point that orchestrates training and evaluation"""

from enviroment import Environment
from agents import QLearningAgent
from training import Trainer, Evaluator
from metrics.visualization import create_all_visualizations
from mdp import GoalReachedReward, StepPenaltyReward, CollisionPenaltyReward, RandomObstacleEvent
from config import base_config, experiement_config as exp_config


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

    # Create agent with default parameters
    agent = QLearningAgent(
        n_states=base_config.N_STATES,
        n_actions=base_config.N_ACTIONS,
        alpha=exp_config.ALPHA,
        gamma=exp_config.GAMMA,
    )

    # Create trainer and evaluator
    trainer = Trainer()
    evaluator = Evaluator()

    # Train agent
    print("Training agent...")
    trained_agent, metrics_logger = trainer.train(
        agent=agent,
        env=env,
        n_episodes=exp_config.N_EPISODES,
        epsilon_start=exp_config.EPSILON_START,
        epsilon_end=exp_config.EPSILON_END,
        epsilon_decay=exp_config.EPSILON_DECAY,
        silent=False,
    )

    # Evaluate agent with random baseline comparison
    print("Evaluating agents...")
    metrics_logger, trained_results, random_results = evaluator.evaluate_with_random_baseline(
        agent=trained_agent, logger=metrics_logger, env=env, n_episodes=base_config.N_EVAL_EPISODES
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_all_visualizations(
        metrics_logger.get_experiment_dir(), trained_results, random_results, trained_agent, env
    )

    print("\n" + "=" * 60)
    print(" TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
