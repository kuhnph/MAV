# Flight Dynamics

Run the viewer from the repository root:

```bash
python flightDynamics/main.py
```

Choose an experiment by editing `EXPERIMENT` near the top of
`flightDynamics/main.py`.

- `open_loop`: starts from straight-and-level trim and accepts keyboard input.
- `trim`: edit airspeed, climb angle, and turn radius in
  `flightDynamics/experiments/trim.py`.
- `control_surfaces`: edit elevator, throttle, aileron, and rudder in
  `flightDynamics/experiments/control_surfaces.py`.

Only `open_loop` accepts flight-control keys:

- Up/down: elevator
- Left/right: aileron
- A/D: rudder
- Space/X: increase/decrease throttle
