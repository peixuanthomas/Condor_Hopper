# Condor Hopper

`Condor_Hopper` controls the active KSP vessel through kRPC for a vertical 15 km hop, engine cutoff and ballistic coast through apogee, LCvx-guided relight, and return to the launch point.

## Commands

```powershell
python main.py probe
python main.py simulate --target-altitude 15000
python main.py fly --target-altitude 15000 --execute
```

`probe` is read-only. `fly` refuses to stage or ignite unless `--execute` is present.

Flight and simulation artifacts are written below `flight_logs/YYYY-MM-DD_Condor_Hopper/`. Plot a telemetry file with:

```powershell
python analyze_flight.py flight_logs\YYYY-MM-DD_Condor_Hopper\telemetry_HHMMSS.csv
```

## Control architecture

- Three-degree-of-freedom LCvx/SOCP guidance with log mass, thrust-magnitude slack, thrust cone, glide-slope, fuel reserve, and terminal constraints.
- CLARABEL trajectory solves with fixed-duration outer searches for ascent duration and landing ignition.
- 20 Hz feed-forward plus position/velocity feedback tracking.
- Ascent uses SAS stability assist by itself to suppress high-centre-of-mass wobble. At 15 km SAS is released before kRPC direction control follows LCvx descent vectors, so the two controllers never fight. RCS is enabled only after liftoff; grid fins are enabled through brakes only after confirmed descent; initially retracted legs deploy only for terminal landing.

The exact SpaceX flight software is not public. This project implements the public lossless-convexification/G-FOLD mathematical approach rather than claiming an exact proprietary reproduction.
