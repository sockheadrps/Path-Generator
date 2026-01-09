# Path Generator
![Tests](tests/tests-badge.svg) ![Coverage](tests/coverage-badge.svg)

A Python library for generating human-like mouse paths using PD (Proportional-Derivative) control.

![](path_viewer.gif)

## Installation

### Core Library
Install the core library (only requires numpy):
```bash
pip install pathgenerator
```

### With Windows Emulator
Install with `pywin32` for high-performance cursor emulation (Windows only):
```bash
pip install pathgenerator[windows]
```

### With Server
Install with the optional FastAPI server:
```bash
pip install pathgenerator[server]
```

## Usage

### Human Motion Parameters
- `mouse_velocity`: Base movement velocity (unitless, 0.1-1.0).
- `kp_start` / `kp_end`: Correction strength at start vs. end.
- `stabilization`: Damping factor to smooth out jitters (0.0-1.0).
- `noise`: Amount of random hand tremor (0.0-10.0+).
- `overshoot_prob`: Probability of overshooting the target and correcting (0.0-1.0).
- `arc_strength`: Tendency to move in a curved arc (0.0-0.5).
- `keep_prob_start` / `keep_prob_end`: Point density (drop probability) at start/end.
- `offset_x` / `offset_y`: Global offset added to all points.

### Generating Paths
```python
from pathgenerator import PDPathGenerator

gen = PDPathGenerator()
path, progress, steps, params = gen.generate_path(
    start_x=100.0, start_y=100.0,
    end_x=1800.0, end_y=900.0,
    mouse_velocity=0.65,
    noise=2.6,
    offset_x=0, offset_y=0  # Optional viewport offset
)
```



### Tuning Workflow (Server + JSON)

The easiest way to find realistic parameters is to use the interactive playground.
*(Requires: `pip install pathgenerator[server]`)*

1.  **Launch the server**:
    ```bash
    python -m pathgenerator.server
    ```
2.  **Tune settings** at `http://127.0.0.1:8001`.
3.  **Download JSON**: Click "Export Preset" to save your settings as `human_relaxed.json`.
4.  **Load in Python**:
    ```python
    # Initialize generator with your custom preset
    gen = PDPathGenerator("human_relaxed.json")
    
    # All generated paths will now use those settings by default
    path, *_ = gen.generate_path(100, 100, 500, 500)
    ```

### Executing Paths (Windows Only)
The `PathEmulator` class includes a helper `get_position()` to simplify generating paths from your current mouse location.

```python
from pathgenerator import PDPathGenerator, PathEmulator

# Requires: pip install pathgenerator[windows]
emulator = PathEmulator()
gen = PDPathGenerator()

# 1. Get current mouse position
start_x, start_y = emulator.get_position()

# 2. Generate path to target (e.g., 500, 500)
# calculate offset if you are targeting a window relative to screen 0,0
path, *_ = gen.generate_path(start_x, start_y, 500, 500)

# 3. Move the mouse (optional delay to control playback speed)
emulator.execute_path(path, delay_between_points=0.001)
```
