"""TenFlix package: V3 benchmark interfaces and the V4 live engine."""

from .config import load_config
from .pipeline import prepare_data, train_run
from .recommender import HybridRecommender
from .types import Recommendation

__all__ = ["HybridRecommender", "Recommendation", "load_config", "prepare_data", "train_run"]
__version__ = "4.0.0"
