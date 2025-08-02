import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
import mujoco
import mujoco.viewer

# Ensure packages are discoverable
sys.path.append('source/berkeley_humanoid_lite')

# Stub out external dependency modules before importing the environment
# Create minimal config and gamepad modules
policy_pkg = types.ModuleType('berkeley_humanoid_lite_lowlevel.policy')
config_mod = types.ModuleType('berkeley_humanoid_lite_lowlevel.policy.config')
gamepad_mod = types.ModuleType('berkeley_humanoid_lite_lowlevel.policy.gamepad')

@dataclass
class Cfg:
    num_joints: int
    physics_dt: float
    policy_dt: float
    joint_kp: list
    joint_kd: list
    effort_limits: list
    action_indices: list
    default_base_position: list
    default_joint_positions: list

class DummyGamepad:
    def __init__(self):
        self.commands = {
            "mode_switch": 0.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
        }

    def run(self):
        pass

config_mod.Cfg = Cfg
gamepad_mod.Se2Gamepad = DummyGamepad
policy_pkg.config = config_mod
policy_pkg.gamepad = gamepad_mod

sys.modules['berkeley_humanoid_lite_lowlevel'] = types.ModuleType('berkeley_humanoid_lite_lowlevel')
sys.modules['berkeley_humanoid_lite_lowlevel.policy'] = policy_pkg
sys.modules['berkeley_humanoid_lite_lowlevel.policy.config'] = config_mod
sys.modules['berkeley_humanoid_lite_lowlevel.policy.gamepad'] = gamepad_mod

# Patch MuJoCo's model loader and viewer to avoid external assets and GUI
xml_model = """
<mujoco>
  <option timestep="0.01" />
  <worldbody>
    <body name="base">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
      <body name="link">
        <joint name="hinge" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge"/>
  </actuator>
</mujoco>
"""

def _dummy_from_xml_path(path):
    return mujoco.MjModel.from_xml_string(xml_model)

class _DummyViewer:
    def sync(self):
        pass

mujoco.MjModel.from_xml_path = staticmethod(_dummy_from_xml_path)
mujoco.viewer.launch_passive = lambda model, data: _DummyViewer()

# Import the environment module after stubbing dependencies
from berkeley_humanoid_lite.environments.mujoco import MujocoSimulator

# Simplify observations to avoid dependency on sensors or command input
MujocoSimulator._get_observations = lambda self: torch.zeros(0)


def test_reset_sets_joint_states():
    cfg = Cfg(
        num_joints=1,
        physics_dt=0.01,
        policy_dt=0.02,
        joint_kp=[0.0],
        joint_kd=[0.0],
        effort_limits=[1.0],
        action_indices=[0],
        default_base_position=[0.1, 0.2, 0.3],
        default_joint_positions=[0.5],
    )

    sim = MujocoSimulator(cfg)
    sim.reset()

    np.testing.assert_allclose(sim.mj_data.qpos[0:3], cfg.default_base_position)
    np.testing.assert_allclose(sim.mj_data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(sim.mj_data.qpos[7:8], cfg.default_joint_positions)
    assert np.allclose(sim.mj_data.qvel, 0.0)
