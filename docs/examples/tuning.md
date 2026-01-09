# Tuning Parameters

This guide explains how to tune path generation parameters for different behaviors.

## Parameter Overview

| Category | Parameters | Purpose |
|----------|------------|---------|
| **Speed & Motion** | `mouse_velocity`, `stabilization` | Control movement speed and smoothness |
| **Correction** | `kp_start`, `kp_end` | How aggressively path steers to target |
| **Character** | `noise`, `arc_strength`, `overshoot_prob` | Human-like imperfections |
| **Density** | `keep_prob_start`, `keep_prob_end` | How many points in final path |

### 1. Speed & Fluidity

- **`mouse_velocity`**: Base movement velocity (unitless, ~0.1 to 1.0).
  - Higher = Faster movement.
  - *Typical*: `0.3` to `0.8`.

| Value | Effect |
|-------|--------|
| 0.15 | Slow, deliberate |
| 0.35 | Natural (default) |
| 0.50 | Quick, rushed |

```python
# Slow, careful movement
path, *_ = gen.generate_path(100, 200, 500, 400, mouse_velocity=0.15)

# Quick snap
path, *_ = gen.generate_path(100, 200, 500, 400, mouse_velocity=0.50)
```

!!! note
    The actual velocity is modified by Fitts's Law braking - movement slows near the target regardless of speed setting.

## Correction Strength (kp_start, kp_end)

Controls how aggressively the path corrects toward the target.

- **kp_start**: Correction in the first half of the path
- **kp_end**: Correction in the second half (fine-tuning)

```python
# Low correction = more curved, natural paths
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    kp_start=0.005,
    kp_end=0.005
)

# High end correction = precise target acquisition
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    kp_start=0.01,
    kp_end=0.025
)
```

# Example: High Precision
path, *_ = gen.generate_path(
    100, 100, 500, 500,
    mouse_velocity=0.25,
    kp_end=0.03,
    noise=0.05
)

!!! tip "Typical Settings"
    - **Gaming/fast**: Low kp_start, moderate kp_end
    - **Form filling**: Moderate both
    - **Drawing**: Low both for smooth curves

## Stabilization

Smooths out velocity changes. Higher = smoother but less responsive.

| Value | Effect |
|-------|--------|
| 0.0 | Raw, jittery |
| 0.15 | Natural (default) |
| 0.4 | Very smooth, flowing |

```python
# Smooth, flowing curves
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    stabilization=0.4
)
```

## Noise

Simulates hand tremor and micro-imprecision using correlated (Ornstein-Uhlenbeck) noise.

| Value | Effect |
|-------|--------|
| 0.0 | Perfect, robotic |
| 0.2 | Slight human wobble |
| 0.5 | Noticeable tremor |
| 1.0 | Very shaky |

```python
# Add natural hand tremor
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    noise=0.25
)
```

!!! info "Noise Decay"
    Noise automatically reduces near the target to ensure accurate endpoint arrival.

## Arc Strength

Makes the path curve in an arc (like a slight bow) rather than going straight.

| Value | Effect |
|-------|--------|
| 0.0 | Straight line tendency |
| 0.15 | Subtle arc |
| 0.3 | Noticeable curve |

```python
# Curved path
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    arc_strength=0.2,
    arc_sign=1  # Curve "up" in unit space
)
```

# Example: Casual / Browsing
path, *_ = gen.generate_path(
    100, 100, 500, 500,
    mouse_velocity=0.5,
    arc_strength=0.2,
    noise=1.5,
    arc_sign=1  # Curve "up" in unit space
)

!!! tip "Arc Direction"
    Use `arc_sign=1` or `arc_sign=-1` to control curve direction. Leave as `None` for random.

## Overshoot

Probability of overshooting the target and correcting back.

```python
# Sometimes overshoot (30% chance)
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    overshoot_prob=0.3
)
```

# Example: Flick Shot / Panic
path, *_ = gen.generate_path(
    100, 100, 500, 500,
    mouse_velocity=0.9,
    overshoot_prob=0.8
    # error_limit not supported in current version
)

## Point Density

Controls how many points are kept in the final path.

- **keep_prob_start**: Probability of keeping each point at path start
- **keep_prob_end**: Probability at path end

```python
# Sparse start, dense end (natural - fast then slow)
path, *_ = gen.generate_path(
    100, 200, 500, 400,
    keep_prob_start=0.5,  # Skip ~50% of early points
    keep_prob_end=0.99    # Keep nearly all late points
)
```

## Variance

Adds random variation to all other parameters for more natural variety.

```python
# Same settings, but each path is slightly different
for _ in range(5):
    path, *_ = gen.generate_path(
        100, 200, 500, 400,
        mouse_velocity=0.35,
        noise=0.2,
        variance=0.15  # ±15% variation on all params
    )
```

---

## Presets

### Robotic/Precise
```python
path, *_ = gen.generate_path(
    start_x, start_y, end_x, end_y,
    mouse_velocity=0.3,
    kp_start=0.02,
    kp_end=0.02,
    stabilization=0.1,
    noise=0.0,
    arc_strength=0.0
)
```

### Natural Human
```python
path, *_ = gen.generate_path(
    start_x, start_y, end_x, end_y,
    mouse_velocity=0.35,
    kp_start=0.012,
    kp_end=0.008,
    stabilization=0.15,
    noise=0.25,
    arc_strength=0.12,
    variance=0.1
)
```

### Rushed/Sloppy
```python
path, *_ = gen.generate_path(
    start_x, start_y, end_x, end_y,
    mouse_velocity=0.5,
    kp_start=0.008,
    kp_end=0.015,
    stabilization=0.1,
    noise=0.4,
    arc_strength=0.25,
    overshoot_prob=0.25,
    variance=0.2
)
```

### Elderly/Careful
```python
path, *_ = gen.generate_path(
    start_x, start_y, end_x, end_y,
    mouse_velocity=0.2,
    kp_start=0.015,
    kp_end=0.01,
    stabilization=0.35,
    noise=0.35,  # More tremor
    arc_strength=0.1,
    variance=0.15
)
```
