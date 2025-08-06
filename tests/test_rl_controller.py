import sys
sys.modules.pop('berkeley_humanoid_lite_lowlevel', None)
sys.modules.pop('berkeley_humanoid_lite_lowlevel.policy', None)
sys.path.append('source')

import numpy as np
from berkeley_humanoid_lite_lowlevel.policy import RlController


def test_rl_controller_act_and_reset():
    controller = RlController(obs_dim=3, action_dim=2)
    obs = np.zeros(3, dtype=np.float32)
    action = controller.act(obs)
    assert action.shape == (2,)
    assert controller.last_action.shape == (2,)
    controller.reset()
    assert np.allclose(controller.last_action, 0.0)
