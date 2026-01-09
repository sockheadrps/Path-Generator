# Path Generator
![Tests](tests/tests-badge.svg) ![Coverage](tests/coverage-badge.svg)

A Python library for generating human-like mouse paths using PD (Proportional-Derivative) control.

![](path_viewer.gif)

## Installation

### Core Library
Install the core library (only requires numpy):
```bash
pip install git+https://github.com/sockheadrps/Path-Generator.git
```

### With Windows Emulator
Install with `pywin32` for high-performance cursor emulation (Windows only):
```bash
pip install "pathgenerator[windows] @ git+https://github.com/sockheadrps/Path-Generator.git"
```

### With Server
Install with the optional FastAPI server:
```bash
pip install "pathgenerator[server] @ git+https://github.com/sockheadrps/Path-Generator.git"
```

## Usage

### Human Motion Parameters
- `mouse_velocity`: Base movement velocity (unitless, 0.1-1.0).
- `kp_start`/`kp_end`: PD controller correction strength.

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

### Executing Paths (Windows Only)
Use the `PathEmulator` to execute paths using `win32api`.

```python
from pathgenerator import PDPathGenerator, PathEmulator

# Initialize emulator (requires pip install pathgenerator[windows])
emulator = PathEmulator()

# Get current mouse position
start_x, start_y = emulator.get_position()

gen = PDPathGenerator()
path, *_ = gen.generate_path(start_x, start_y, 500, 500)

# Execute path
emulator.execute_path(path, delay_between_points=0.01)
```

### Server
To run the playground server:
```bash
python -m pathgenerator.server
```
Visit http://127.0.0.1:8001 in your browser.
