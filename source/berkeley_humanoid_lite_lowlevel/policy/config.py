"""Minimal configuration dataclass for low-level policies."""

from dataclasses import dataclass


@dataclass
class Cfg:
    """Configuration for Mujoco simulation and controllers.

    Attributes:
        num_joints: Number of actuated joints.
        policy_dt: Time delta between policy updates.
        physics_dt: Time delta of the physics simulation.
    """

    num_joints: int = 22
    policy_dt: float = 0.02
    physics_dt: float = 0.002
