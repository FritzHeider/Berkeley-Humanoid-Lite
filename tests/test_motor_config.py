import json
import math
from pathlib import Path


def _check_finite(value):
    if isinstance(value, dict):
        for v in value.values():
            _check_finite(v)
    elif isinstance(value, list):
        for v in value:
            _check_finite(v)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value), "Non-finite number in motor configuration"


def test_motor_configuration_all_numbers_finite():
    cfg_path = Path('motor_configuration.json')
    data = json.loads(cfg_path.read_text())
    _check_finite(data)
