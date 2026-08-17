# Flight Dynamics

Run the viewer from the repository root:

```bash
python flightDynamics/main.py
```

Choose a scenario by editing `SCENARIO` near the top of
`flightDynamics/main.py`.

- `open_loop`: starts from straight-and-level trim and accepts keyboard input.
- `trim`: edit airspeed, climb angle, and turn radius in
  `flightDynamics/scenarios/trim.py`.
- `control_surfaces`: edit elevator, throttle, aileron, and rudder in
  `flightDynamics/scenarios/control_surfaces.py`.

Only `open_loop` accepts flight-control keys:

- Up/down: elevator
- Left/right: aileron
- A/D: rudder
- Space/X: increase/decrease throttle
