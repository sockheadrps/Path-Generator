# Basic Usage

This guide covers common usage patterns for the path generator.

## Simple Path Generation

```python
from pathgenerator import PDPathGenerator

gen = PDPathGenerator()

path, progress, steps, params = gen.generate_path(
    start_x=100, start_y=200,
    end_x=500, end_y=400
)
```

### Return Values

| Value | Type | Description |
|-------|------|-------------|
| `path` | `np.ndarray` | Array of (x, y) coordinates |
| `progress` | `List[float]` | Progress value (0-1) for each point |
| `steps` | `int` | Number of simulation steps |
| `params` | `dict` | Actual parameters used (after randomization) |

## Moving the Mouse

To actually move the mouse, pair this with a mouse control library, or (windows only) use the optional `PathEmulator` included when installing with `pip install pathgenerator[windows]`

=== "PathEmulator (Windows)"

    Recommended for Windows users for high performance and smooth movement.

    ```python
    from pathgenerator import PDPathGenerator, PathEmulator

    # Requires: pip install pathgenerator[windows]
    emulator = PathEmulator()
    gen = PDPathGenerator()

    # Generate from current mouse position
    start_x, start_y = emulator.get_position()
    path, *_ = gen.generate_path(start_x, start_y, 500, 400)

    # Execute
    emulator.execute_path(path)
    ```

=== "pyautogui"

    ```python
    import pyautogui
    import time
    from pathgenerator import PDPathGenerator

    pyautogui.PAUSE = 0  # Disable default 0.1s pause between actions

    gen = PDPathGenerator()
    path, *_ = gen.generate_path(100, 200, 500, 400)

    for x, y in path:
        pyautogui.moveTo(x, y)
        time.sleep(0.01)
    ```

=== "pynput"

    ```python
    from pynput.mouse import Controller
    import time
    from pathgenerator import PDPathGenerator

    mouse = Controller()
    gen = PDPathGenerator()
    path, *_ = gen.generate_path(100, 200, 500, 400)

    for x, y in path:
        mouse.position = (x, y)
        time.sleep(0.01)
    ```

## Canvas Size

The generator is resolution-independent, but you can specify your canvas/viewport size for optimal step sizing:

```python
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    canvas_width=2560,
    canvas_height=1440
)
```

!!! tip "Default Size"
    The default is 1920x1080. Specifying your actual canvas size helps with step size calculations. The larger dimension is used for scaling.

## Window/Viewport Targeting

When targeting a specific window on screen, use `offset_x` and `offset_y` to translate coordinates:

```python
# Window at position (200, 100) on screen, size 800x600
# Move from (50, 50) to (150, 100) within that window

path, *_ = gen.generate_path(
    50, 50, 150, 100,           # Coordinates relative to window
    canvas_width=800,           # Window dimensions
    canvas_height=600,
    offset_x=200,               # Window's X position on screen
    offset_y=100                # Window's Y position on screen
)

# Output path is in screen coordinates: (250, 150) → (350, 200)
```

The offset is applied to all output coordinates, so you can think in window-relative terms while getting screen-ready output.

## Using Presets

You can load motion parameters (speed, noise, etc.) from a JSON file:

```json
// natural.json
{
  "mouse_velocity": 0.65,
  "kp_start": 0.0004,
  "kp_end": 0.0004,
  "stabilization": 0.29,
  "noise": 2.6,
  "keep_prob_start": 0.7,
  "keep_prob_end": 0.98,
  "arc_strength": 0.27,
  "variance": 0.45,
  "overshoot_prob": 0.45
}
```

Initialize the generator with the file path:

```python
gen = PDPathGenerator('natural.json')

# Uses values from the JSON file
path, *_ = gen.generate_path(100, 200, 500, 400)

# You can still override specific values
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    mouse_velocity=0.8  # Overrides preset velocity
)
```
