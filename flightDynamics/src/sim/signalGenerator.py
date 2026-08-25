import numpy as np

class SignalGenerator:
    def __init__(self):
        pass

    def square(self, time, amplitude, frequency):
        A = amplitude
        t = time
        f = frequency
        return A * np.sign(np.sin(2 * np.pi * f * t))