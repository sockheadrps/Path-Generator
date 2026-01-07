# Path Generator

A Python library for generating human-like mouse movement paths using PD (Proportional-Derivative) control.

## Features

- **Human-like trajectories** - Curved paths with natural velocity profiles
- **Fitts's Law compliance** - Automatic deceleration near targets
- **Configurable behavior** - Tune speed, curvature, noise, and more
- **Resolution independent** - Works across different screen sizes
- **Overshoot simulation** - Optional target overshoot and correction

## Quick Start

```python
from pathgenerator import PDPathGenerator

# Create generator
gen = PDPathGenerator()

# Generate a path from (100, 200) to (500, 400)
path, progress, steps, params = gen.generate_path(
    start_x=100, start_y=200,
    end_x=500, end_y=400,
    speed=0.35,
    noise=0.2,
    arc_strength=0.15
)

# path is a numpy array of (x, y) coordinates
for x, y in path:
    print(f"Move to ({x:.1f}, {y:.1f})")
```

## Installation

```bash
pip install numpy
```

Then add the `pathgenerator` package to your project.

## How It Works

The generator uses a **unit-frame approach**:

1. Transform the problem so start=(0,0) and target=(1,0)
2. Simulate movement with PD control for correction
3. Add human-like noise and velocity profiles
4. Transform back to screen coordinates

See the [Algorithm Guide](path_generation.md) for a detailed explanation.

## Example Paths

### Natural Movement
```python
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    speed=0.35,
    kp_start=0.015,
    kp_end=0.008,
    noise=0.3,
    arc_strength=0.15
)
```

### Quick/Sloppy Movement
```python
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    speed=0.5,
    noise=0.5,
    arc_strength=0.25,
    overshoot_prob=0.3
)
```

### Precise/Careful Movement
```python
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    speed=0.2,
    kp_end=0.02,
    stabilization=0.3,
    noise=0.1
)
```

## Next Steps

- [Algorithm Guide](path_generation.md) - Understand how path generation works
- [Basic Usage](examples/basic_usage.md) - Common usage patterns
- [Tuning Parameters](examples/tuning.md) - Fine-tune path characteristics
- [API Reference](api/generator.md) - Full API documentation
