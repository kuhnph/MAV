"""Read attitude commands without blocking the simulation thread."""

from queue import Empty, Queue
from threading import Thread
import sys

import numpy as np


class TerminalCommands:
    names = ("roll", "pitch", "yaw")

    def __init__(self):
        self.pending = Queue()

    def start(self):
        print("Terminal commands (degrees): roll=0, pitch=1, yaw=0. Press Enter to apply.")
        Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in sys.stdin:
            self.submit(line)

    def submit(self, line):
        name, separator, value = line.strip().partition("=")
        name = name.strip()
        try:
            if not separator or name not in self.names:
                raise ValueError
            degrees = float(value)
            radians = np.deg2rad(degrees)
            if not np.isfinite(radians):
                raise ValueError
        except ValueError:
            print("Use roll=<degrees>, pitch=<degrees>, or yaw=<degrees> (finite numbers only).")
            return
        self.pending.put((self.names.index(name), radians, degrees))

    def apply(self, eta_command):
        """Apply queued changes on the simulation thread; retain other commands."""
        while True:
            try:
                index, radians, degrees = self.pending.get_nowait()
            except Empty:
                return
            eta_command[index, 0] = radians
            print(f"Applied {self.names[index]}={degrees:g} degrees")
