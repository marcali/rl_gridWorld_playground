"""Training package for RL agents"""

from .trainer import Trainer
from .evaluator import Evaluator


# For backward compatibility, create function-based interface
def train(*args, **kwargs):
    """Train an agent - backward compatibility function"""

    trainer = Trainer()
    return trainer.train(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Evaluate an agent - backward compatibility function"""
    evaluator = Evaluator()
    return evaluator.evaluate(*args, **kwargs)


def evaluate_with_random_baseline(*args, **kwargs):
    """Evaluate agent with random baseline - backward compatibility function"""
    evaluator = Evaluator()
    return evaluator.evaluate_with_random_baseline(*args, **kwargs)


__all__ = ["Trainer", "Evaluator", "train", "evaluate", "evaluate_with_random_baseline"]
