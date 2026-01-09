import numpy as np
import pytest
from pathgenerator.geometry import get_unit_transform, rotate_scale_path_to_hit_target

# -----------------------------------------------------------------------------
# Tests for get_unit_transform
# -----------------------------------------------------------------------------

def test_unit_transform_horizontal():
    """Test the simplest case: a horizontal line.
    
    If start=(0,0) and target=(100,0), the vector is already aligned with X.
    - Distance should be 100.
    - Rotation should be Identity (no rotation needed).
    """
    start = np.array([0, 0])
    target = np.array([100, 0])
    
    R, D = get_unit_transform(start, target)
    
    assert D == 100.0
    # Expected: Identity Matrix [[1, 0], [0, 1]]
    expected_R = np.eye(2)
    assert np.allclose(R, expected_R)

def test_unit_transform_diagonal_345():
    """Test a 3-4-5 triangle case (Classic Pythagorean triple).
    
    Start=(0,0), Target=(300, 400).
    - Distance should be 500.
    - Matrix R should rotate the vector (300, 400) to align with (500, 0).
    """
    start = np.array([0, 0])
    target = np.array([300, 400])
    
    R, D = get_unit_transform(start, target)
    
    # 1. Check Distance
    assert np.isclose(D, 500.0)
    
    # 2. Check the Transform Logic
    # If we apply R to the vector (target-start), we should get [Distance, 0]
    vector = target - start
    rotated_vector = vector @ R.T  # Note: The function builds R for v @ R.T style multiplication
    
    expected_vector = np.array([500.0, 0.0])
    assert np.allclose(rotated_vector, expected_vector, atol=1e-5)

def test_unit_transform_zero_distance():
    """Test safety check for Start == Target."""
    start = np.array([50, 50])
    target = np.array([50, 50])
    
    # Should not crash (ZeroDivisionError)
    R, D = get_unit_transform(start, target)
    
    # The code forces D=1.0 if distance is 0 to avoid division
    assert D == 1.0
    # Since vector is [0,0], the rotation matrix becomes zeros in the current implementation
    assert np.allclose(R, np.zeros((2,2)))

# -----------------------------------------------------------------------------
# Tests for rotate_scale_path_to_hit_target
# -----------------------------------------------------------------------------

def test_fix_path_already_perfect():
    """If the path already hits the target, it should remain unchanged."""
    start = (0, 0)
    target = (100, 0)
    # A path that goes straight to target
    path = np.array([[0, 0], [50, 0], [100, 0]], dtype=float)
    
    fixed_path = rotate_scale_path_to_hit_target(path, start, target)
    
    assert np.allclose(path, fixed_path)

def test_fix_path_rotation_only():
    """Test correcting a path that missed by angle (Rotation Correction)."""
    start = (0, 0)
    target = (0, 100) # Target is straight UP (90 degrees)
    
    # The generated path went straight RIGHT (0 degrees)
    # Length is correct (100), but angle is wrong by 90 degrees.
    path = np.array([[0, 0], [50, 0], [100, 0]], dtype=float)
    
    # We disable scaling to test rotation in isolation
    fixed_path = rotate_scale_path_to_hit_target(path, start, target, scale_to_distance=False)
    
    # The last point should now match the target (0, 100)
    assert np.allclose(fixed_path[-1], [0, 100], atol=1e-5)
    
    # The midpoint (50, 0) should have rotated to (0, 50)
    assert np.allclose(fixed_path[1], [0, 50], atol=1e-5)

def test_fix_path_scaling_only():
    """Test correcting a path that fell short (Scaling Correction)."""
    start = (0, 0)
    target = (100, 0)
    
    # Path goes in correct direction but stops at 50 (Halfway)
    path = np.array([[0, 0], [25, 0], [50, 0]], dtype=float)
    
    fixed_path = rotate_scale_path_to_hit_target(path, start, target, scale_to_distance=True)
    
    # Endpoint should be scaled to 100
    assert np.allclose(fixed_path[-1], [100, 0])
    # Midpoint should be scaled to 50
    assert np.allclose(fixed_path[1], [50, 0])

def test_fix_path_mixed_correction():
    """Test correcting a path that missed both distance AND angle."""
    start = (0, 0)
    target = (100, 100) # Target is at 45 degrees, dist ~141.4
    
    # Path went straight right to (100, 0)
    path = np.array([[0, 0], [100, 0]], dtype=float)
    
    fixed_path = rotate_scale_path_to_hit_target(path, start, target)
    
    # Should snap to target
    assert np.allclose(fixed_path[-1], [100, 100])
    
    # Start point should remain locked
    assert np.allclose(fixed_path[0], [0, 0])

def test_fix_path_degenerate_cases():
    """Test behavior when start/end points overlap (avoid division by zero)."""
    start = (0, 0)
    target = (10, 10)
    
    # Case 1: Path has 0 length (All points are start)
    path = np.array([[0, 0], [0, 0]], dtype=float)
    
    # Should return original path (degenerate)
    fixed = rotate_scale_path_to_hit_target(path, start, target)
    assert np.allclose(fixed, path)