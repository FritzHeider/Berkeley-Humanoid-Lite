"""Placeholder gamepad interface."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Se2Gamepad:
    """Simple representation of an SE(2) gamepad.

    The stub exposes a ``read`` method that returns zero velocities.
    """

    def read(self) -> Tuple[float, float, float]:
        """Return zero translational and angular velocity commands."""
        return (0.0, 0.0, 0.0)
