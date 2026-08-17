from dataclasses import dataclass
from collections.abc import Callable
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
    control_update: Callable[
        [float, np.ndarray, np.ndarray],
        np.ndarray,
    ] | None = None
    desired: np.ndarray | None = None
