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
    pip install git+https://github.com/sockheadrps/Path-Generator.git
    ```

=== "Server/Playground"
    ```bash
    pip install "pathgenerator[server] @ git+https://github.com/sockheadrps/Path-Generator.git"
    ```

=== "Windows Emulation"
    ```bash
    pip install "pathgenerator[windows] @ git+https://github.com/sockheadrps/Path-Generator.git"
    ```

=== "All Extras"
    ```bash
    pip install "pathgenerator[server,windows] @ git+https://github.com/sockheadrps/Path-Generator.git"
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

### Path Execution (Windows)

Use the optional `PathEmulator` to move the mouse cursor:

```python
from pathgenerator import PDPathGenerator, PathEmulator

# Requires: pip install pathgenerator[windows]
emulator = PathEmulator()
gen = PDPathGenerator()

# Generate from current mouse position
start_x, start_y = emulator.get_position()
path, *_ = gen.generate_path(start_x, start_y, 500, 500)

emulator.execute_path(path)
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
