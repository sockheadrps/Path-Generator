# Path Generator

A Python library for generating human-like mouse movement paths using PD (Proportional-Derivative) control.

## Features

- **Human-like trajectories** - Curved paths with natural velocity profiles
- **Fitts's Law compliance** - Automatic deceleration near targets
- **Configurable behavior** - Tune speed, curvature, noise, and more
- **Resolution independent** - Works across different screen sizes
- **Overshoot simulation** - Optional target overshoot and correction


## Installation

=== "Base"
    ```bash
    pip install pathgenerator
    ```

=== "Server/Playground"
    ```bash
    pip install pathgenerator[server]
    ```

=== "Windows Emulation"
    ```bash
    pip install pathgenerator[windows]
    ```

=== "All Extras"
    ```bash
    pip install pathgenerator[server,windows]
    ```

## Quick Start

```python
from pathgenerator import PDPathGenerator

# Create generator
gen = PDPathGenerator()
# gen = PDPathGenerator('params.json') to load params from a json file

# Define points
start_x, start_y = 100, 200
end_x, end_y = 900, 1000

# Generate a path
path, prog_list, steps, params = gen.generate_path(
    start_x=start_x, start_y=start_y,
    end_x=end_x, end_y=end_y,
    offset_x=0, offset_y=0  # Optional offset if you target a relative area
)


# path is a numpy array of (x, y) coordinates
for x, y in path:
    print(f"Move to ({x:.1f}, {y:.1f})")
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


## How It Works

The generator uses a **unit-frame approach**:

1. Transform the problem so start=(0,0) and target=(1,0)
2. Simulate movement with PD control for correction
3. Add human-like noise and velocity profiles
4. Transform back to screen coordinates

See the [Algorithm Guide](path_generation.md) for a detailed explanation.

## Next Steps

- [Algorithm Guide](path_generation.md) - Understand how path generation works
- [Basic Usage](examples/basic_usage.md) - Common usage patterns
- [Tuning Parameters](examples/tuning.md) - Fine-tune path characteristics
- [API Reference](api/generator.md) - Full API documentation
