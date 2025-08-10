from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, Light
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos


def randomize_actuator_torque_constant(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_constant_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if torque_constant_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                stiffness = actuator.stiffness.to(asset.device).clone()
                damping = actuator.damping.to(asset.device).clone()
                scale = _randomize_prop_by_op(
                    torch.ones_like(stiffness, device=asset.device),
                    torque_constant_params,
                    env_ids,
                    actuator_joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                stiffness[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]
                damping[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]

                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
):
    """Randomize the color and intensity of a light asset."""

    asset: Light = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if color_range is not None:
        r = torch.rand(len(env_ids), device=asset.device)
        g = torch.rand(len(env_ids), device=asset.device)
        b = torch.rand(len(env_ids), device=asset.device)
        colors = torch.stack(
            [
                r * (color_range[0][1] - color_range[0][0]) + color_range[0][0],
                g * (color_range[1][1] - color_range[1][0]) + color_range[1][0],
                b * (color_range[2][1] - color_range[2][0]) + color_range[2][0],
            ],
            dim=-1,
        )
        if hasattr(asset, "set_color"):
            asset.set_color(colors, env_ids=env_ids)
        elif hasattr(asset, "data") and hasattr(asset.data, "color"):
            asset.data.color[env_ids] = colors

    if intensity_range is not None:
        intensities = (
            torch.rand(len(env_ids), device=asset.device) * (intensity_range[1] - intensity_range[0])
            + intensity_range[0]
        )
        if hasattr(asset, "set_intensity"):
            asset.set_intensity(intensities, env_ids=env_ids)
        elif hasattr(asset, "data") and hasattr(asset.data, "intensity"):
            asset.data.intensity[env_ids] = intensities
