from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim.params import Params


@dataclass
class Scenario:
    name: str
    description: str
    parameters: "Params"
    accepts_key_input: bool = False
    controller_enabled: bool = False
    omega_dot_command: np.ndarray | None = None
    omega_command: np.ndarray | None = None
