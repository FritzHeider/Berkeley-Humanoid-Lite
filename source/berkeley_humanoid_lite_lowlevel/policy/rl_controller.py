"""Lightweight neural-network based RL controller.

This module provides a small fully-connected neural network that can be used as
an on-device policy.  The controller either loads weights from a provided file
or behaves as a randomly initialised policy, making it suitable for quick
simulation demos or unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


class RlController:
    """Tiny feed-forward policy used for low-level control.

    Parameters
    ----------
    obs_dim:
        Dimension of the observation vector.
    action_dim:
        Dimension of the action vector produced by the policy.
    model_path:
        Optional path to a PyTorch ``state_dict`` containing the weights for
        the policy network.  If omitted, the network is randomly initialised.
    device:
        The device on which to run the policy.  Defaults to CPU.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        model_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Simple two-layer MLP used as the policy network.
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        ).to(self.device)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)

        # Track the most recent action for optional introspection.
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute an action for a given observation.

        Parameters
        ----------
        obs:
            Observation vector provided by the environment.

        Returns
        -------
        np.ndarray
            The action computed by the policy network.
        """

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = self.policy(obs_tensor).cpu().numpy()
        self.last_action = action
        return action

    def reset(self) -> None:
        """Reset internal controller state."""

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
