"""Coordinate transformation utilities for the path generator.

This module provides functions for transforming coordinates between screen space
and a normalized "unit frame" where path generation is resolution-independent.

The unit frame transforms any start→target pair such that:
- Start point becomes (0, 0)
- Target point becomes (1, 0)
- All distances are normalized relative to the start-target distance
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def get_unit_transform(
    start: np.ndarray, 
    target: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute the rotation matrix and distance for unit-frame transformation.
    
    Calculates the transform needed to map screen coordinates to a normalized
    unit frame where start=(0,0) and target=(1,0).
    
    Args:
        start: Starting point as (x, y) array in screen coordinates.
        target: Target point as (x, y) array in screen coordinates.
    
    Returns:
        Tuple containing:
            - R: 2x2 rotation matrix that aligns the start→target vector with +X axis.
            - D: Distance between start and target (used for scaling).
    
    Example:
        >>> start = np.array([100, 200])
        >>> target = np.array([300, 200])
        >>> R, D = get_unit_transform(start, target)
        >>> D
        200.0
        >>> # R is identity since target is already along +X from start
    
    Note:
        To convert screen → unit: `P_unit = (P_screen - start) @ R.T / D`
        To convert unit → screen: `P_screen = start + (P_unit * D) @ R`
    """
    sx, sy = start
    tx, ty = target
    v = np.array([tx - sx, ty - sy], dtype=np.float32)
    D = float(np.hypot(v[0], v[1])) or 1.0
    v /= D
    c, s = v[0], v[1]
    # Rotate by -theta (align vector to x-axis)
    # Applied as: [x, y] @ R.T = [c*x + s*y, -s*x + c*y]
    R = np.array([[c, s],
                  [-s, c]], dtype=np.float32)
    return R, D


def rotate_scale_path_to_hit_target(
    path_xy: np.ndarray, 
    start_xy: Tuple[float, float], 
    target_xy: Tuple[float, float], 
    *, 
    scale_to_distance: bool = True
) -> np.ndarray:
    """Rotate and scale a path so its endpoint exactly matches the target.
    
    After path simulation, there may be small numerical errors that cause the
    final point to not exactly hit the target. This function applies a rotation
    (and optionally uniform scaling) around the start point to correct this.
    
    Args:
        path_xy: Path as (N, 2) numpy array of screen coordinates.
        start_xy: Start point (should match path_xy[0]).
        target_xy: Desired endpoint for the path.
        scale_to_distance: If True, uniformly scale the path so the endpoint
            distance matches exactly. If False, only rotate (preserves path length).
    
    Returns:
        Transformed path as (N, 2) numpy array with path[-1] == target_xy.
    
    Example:
        >>> path = np.array([[0, 0], [50, 10], [95, 5]])  # Slightly off-target
        >>> target = (100, 0)
        >>> fixed = rotate_scale_path_to_hit_target(path, (0, 0), target)
        >>> fixed[-1]
        array([100., 0.])
    
    Note:
        If either the current endpoint or target is very close to the start
        (< 1e-6 distance), the path is returned unchanged to avoid division
        by zero.
    """
    P0 = np.asarray(start_xy,  dtype=np.float32)
    PT = np.asarray(target_xy, dtype=np.float32)
    P  = np.asarray(path_xy,   dtype=np.float32)

    v1 = P[-1] - P0             # current end vector (from start -> last)
    v2 = PT   - P0              # desired end vector (from start -> target)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return P  # degenerate—nothing to do

    # angle to rotate v1 into v2 (signed)
    dot = float(np.dot(v1, v2))
    det = float(v1[0]*v2[1] - v1[1]*v2[0])
    theta = np.arctan2(det, dot)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    # optional uniform scale so magnitudes match (pure rotation if False)
    scale = (n2 / n1) if scale_to_distance else 1.0

    # rotate+scale about the start point
    P_centered = P - P0
    P_new = (P_centered @ R.T) * scale + P0

    # by construction, P_new[-1] == target (up to fp rounding); snap it
    P_new[-1] = PT
    return P_new
