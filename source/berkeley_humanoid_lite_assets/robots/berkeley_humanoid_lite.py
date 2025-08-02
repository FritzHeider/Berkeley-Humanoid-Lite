"""Placeholder robot configuration for Berkeley Humanoid Lite.

These definitions provide minimal structure required for tests and examples.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RobotCfg:
    """Minimal robot configuration used by the simulation."""
    name: str
    num_joints: int


# Basic configuration for the full humanoid and the biped variants.
HUMANOID_LITE_CFG = RobotCfg(name="humanoid_lite", num_joints=22)
HUMANOID_LITE_JOINTS: List[str] = [f"joint_{i}" for i in range(HUMANOID_LITE_CFG.num_joints)]

HUMANOID_LITE_BIPED_CFG = RobotCfg(name="humanoid_lite_biped", num_joints=12)
HUMANOID_LITE_LEG_JOINTS: List[str] = [f"leg_joint_{i}" for i in range(HUMANOID_LITE_BIPED_CFG.num_joints)]
