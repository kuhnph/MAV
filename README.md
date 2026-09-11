# Flight Dynamics

Run the viewer from the repository root:

```bash
python main.py
```

Choose a scenario by editing `SCENARIO` near the top of
`main.py`.

- `open_loop`: starts from straight-and-level trim and accepts keyboard input.
- `trim`: edit airspeed, climb angle, and turn radius in
  `flightDynamics/scenarios/trim.py`.
- `control_surfaces`: edit elevator, throttle, aileron, and rudder in
  `flightDynamics/scenarios/control_surfaces.py`.
- `dynamic_inversion`: while running, type `theta_c=1` in the launching
  terminal and press Enter to command a pitch angle of 1 degree. `phi_c` (roll)
  and `psi_c` (yaw) work the same way. Values are absolute angles in degrees;
  each command remains active until changed. The simulation continues while
  you type. Initial commands are set by `eta_command` in `main.py`.

Only `open_loop` accepts flight-control keys:

- Up/down: elevator
- Left/right: aileron
- A/D: rudder
- Space/X: increase/decrease throttle
