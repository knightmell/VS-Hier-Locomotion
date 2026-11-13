"""Locomotion subset of the MIT-licensed SB3/DSRL implementation."""
from pathlib import Path
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.sac import SAC
from stable_baselines3.dsrl import DSRL, HierarchicalRFSDSRL
__version__ = Path(__file__).with_name("version.txt").read_text().strip()
__all__ = ["SAC", "DSRL", "HierarchicalRFSDSRL", "get_system_info"]
