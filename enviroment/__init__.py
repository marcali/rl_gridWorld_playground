"""Environment package for RL environments"""

from .base_enviroment import BaseEnvironment
from .grid_world import GridWorldEnvironment

# For backward compatibility, alias GridWorldEnvironment as Environment
Environment = GridWorldEnvironment

__all__ = ["BaseEnvironment", "GridWorldEnvironment", "Environment"]
