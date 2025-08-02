import os
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch

# Ensure source path on sys.path
sys.path.append(os.path.abspath("source/berkeley_humanoid_lite"))

# Mock missing lowlevel config module
lowlevel_policy_config = ModuleType("berkeley_humanoid_lite_lowlevel.policy.config")

class Cfg:  # minimal stand-in for the expected class
    pass

lowlevel_policy_config.Cfg = Cfg
lowlevel_policy = ModuleType("berkeley_humanoid_lite_lowlevel.policy")
lowlevel_policy.config = lowlevel_policy_config
lowlevel_policy_gamepad = ModuleType("berkeley_humanoid_lite_lowlevel.policy.gamepad")

class Se2Gamepad:
    def run(self):
        pass

lowlevel_policy_gamepad.Se2Gamepad = Se2Gamepad
lowlevel_policy.gamepad = lowlevel_policy_gamepad
lowlevel_pkg = ModuleType("berkeley_humanoid_lite_lowlevel")
lowlevel_pkg.policy = lowlevel_policy
sys.modules["berkeley_humanoid_lite_lowlevel"] = lowlevel_pkg
sys.modules["berkeley_humanoid_lite_lowlevel.policy"] = lowlevel_policy
sys.modules["berkeley_humanoid_lite_lowlevel.policy.config"] = lowlevel_policy_config
sys.modules["berkeley_humanoid_lite_lowlevel.policy.gamepad"] = lowlevel_policy_gamepad

from berkeley_humanoid_lite.environments.mujoco import MujocoSimulator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_apply_actions_gpu():
    class DummyCfg:
        num_joints = 2
        action_indices = [0, 1]

    dummy_env = SimpleNamespace(
        cfg=DummyCfg(),
        joint_kp=torch.ones(2),
        joint_kd=torch.ones(2),
        effort_limits=torch.full((2,), 2.0),
        mj_data=SimpleNamespace(ctrl=np.zeros(2)),
        _get_joint_pos=lambda: torch.zeros(2),
        _get_joint_vel=lambda: torch.zeros(2),
    )

    actions = torch.tensor([0.5, -0.5], device="cuda")
    MujocoSimulator._apply_actions(dummy_env, actions)

    assert np.allclose(dummy_env.mj_data.ctrl, np.array([0.5, -0.5]))
