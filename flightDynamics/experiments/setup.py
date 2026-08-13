from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.params import Params


@dataclass
class Experiment:
    name: str
    description: str
    parameters: "Params"
    accepts_key_input: bool=False
