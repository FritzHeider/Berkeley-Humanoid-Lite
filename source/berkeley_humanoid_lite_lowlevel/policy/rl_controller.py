"""Placeholder reinforcement learning controller."""


class RlController:
    """Minimal stub RL controller used for simulation demos."""

    def __init__(self) -> None:
        self.last_action = []

    def act(self, _obs=None):
        """Return a zero action for a given observation."""
        self.last_action = []
        return self.last_action

    def reset(self) -> None:
        """Reset internal controller state."""
        self.last_action = []
