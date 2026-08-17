"""Run a dynamic inversion controller in the loop."""

from control.dynamic_inversion import controller
from scenarios.setup import Scenario
from sim.params import Params
import numpy as np

omega_c = np.array([[0,0,0]]).T

def build_scenario():
    parameters=Params()
