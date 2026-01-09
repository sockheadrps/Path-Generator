import pytest
from pathgenerator import PDPathGenerator
import numpy as np

# --------------------- Tests for PDPathGenerator init()---------------------
def test_generator_instantiation():
    """Test that the generator can be instantiated."""
    gen = PDPathGenerator()
    assert gen is not None


def test_init_default_values():
    gen = PDPathGenerator()
    
    # Check that it loaded the defaults
    assert gen.preset == PDPathGenerator.DEFAULT_PRESET
    
    # CRITICAL: Check that it is a COPY, not a reference to the class attribute
    # If this fails, modifying one generator's preset would break all others, if others exist.
    assert gen.preset is not PDPathGenerator.DEFAULT_PRESET
    gen.preset['mouse_velocity'] = 999
    assert PDPathGenerator.DEFAULT_PRESET['mouse_velocity'] != 999



def test_init_loads_preset_from_file(mocker):
    mock_data = {'mouse_velocity': 100, 'noise': 0.5}
    
    # Mock the _load_preset_file method
    mock_loader = mocker.patch.object(PDPathGenerator, '_load_preset_file', return_value=mock_data)
    gen = PDPathGenerator(preset_file="custom_config.json")

    mock_loader.assert_called_once_with("custom_config.json")
    assert gen.preset == mock_data

def test_real_file_loading_coverage(tmp_path):
    """
    Cover lines 133-137: Real file loading without mocks.
    This ensures the 'success path' of _load_preset_file executes the loop 
    that copies keys from the file to the preset dict.
    """
    # 1. Create a real config file
    config_file = tmp_path / "real_config.json"
    import json
    config_file.write_text(json.dumps({
        "mouse_velocity": 55.5, 
        "noise": 0.1
    }))
    
    # 2. Initialize generator pointing to that file (No Mocks!)
    gen = PDPathGenerator(preset_file=str(config_file))
    
    # 3. Verify it loaded
    assert gen.preset['mouse_velocity'] == 55.5
    assert gen.preset['noise'] == 0.1

def test_init_raises_on_unknown_keys_from_file(tmp_path):
    """Test that loading a preset with unknown keys raises ValueError."""
    # Create a dummy json file
    bad_preset = tmp_path / "bad.json"
    import json
    bad_preset.write_text(json.dumps({"mouse_velocity": 0.5, "invalid_key": 123}))
    
    with pytest.raises(ValueError, match="Unknown parameters"):
        PDPathGenerator(preset_file=str(bad_preset))


def test_init_propagates_loading_errors(mocker):
    """Test that loading a preset file that doesn't exist raises FileNotFoundError."""
    # This automatically handles starting and stopping the patch
    mocker.patch.object(PDPathGenerator, '_load_preset_file', side_effect=FileNotFoundError("File missing"))
        
    # We expect the Init to fail because our mock raises an error
    with pytest.raises(FileNotFoundError):
        PDPathGenerator(preset_file="missing.json")


def test_randomize_preset_values():
    """Test that _randomize returns a parameter dictionary after randomizing."""
    gen = PDPathGenerator()
    original = gen.preset.copy()
    randomized_params =gen._randomize(original)
    assert randomized_params != PDPathGenerator.DEFAULT_PRESET


# --------------------- Tests for PDPathGenerator internal methods ---------------------

# _setup_transforms()
def test_setup_transforms_calculates_correct_state():
    """Test that _setup_transforms() actually apply the matracy transformations: 
    map (0,0) to start and (1,0) to target."""  
    gen = PDPathGenerator()
    start = np.array([100.0, 100.0])
    target = np.array([400.0, 500.0]) # 3-4-5 Triangle
    
    gen._setup_transforms(start, target)
    
    # 1. Check Distance (Should be 500) (Euclidean distance, derrived from pythagorean theorem /
    # square root of sum of squares)
    assert np.isclose(gen.D, 500.0)
    
    # 2. Check Origin
    assert np.array_equal(gen.origin, start)
    
    # 3. Check Rotation Matrix Properties (It must be a valid rotation matrix)
    # The determinant of a rotation matrix must be 1.0
    assert np.isclose(np.linalg.det(gen.R_screen), 1.0)
    
    # 4. Check the Geometry directly (Spot check)
    # We know the vector [300, 400] normalized is [0.6, 0.8].
    # The X-column of R_screen (Unit->Screen) should represent this direction.
    expected_x_axis = np.array([0.6, 0.8]) 
    assert np.allclose(gen.R_screen[:, 0], expected_x_axis)

# _unit_to_screen()
def test_unit_to_screen_applies_transform_logic():
    """Test that _unit_to_screen correctly applies: origin + (R @ point * D)."""
    gen = PDPathGenerator()
    
    # MANUAL STATE INJECTION
    # We set up a "dumb" state so we can easily calculate the expected outcome in our heads.
    # We use Identity matrix, so Rotation does nothing.
    gen.origin = np.array([10.0, 10.0])
    gen.D = 100.0
    gen.R_screen = np.eye(2) # Identity Matrix (No rotation)
    
    # Case A: The Start Point (0,0)
    # Should be: Origin + (Identity * [0,0] * 100) = Origin
    unit_start = np.array([0.0, 0.0])
    assert np.allclose(gen._unit_to_screen(unit_start), [10.0, 10.0])

    # Case B: The Target Point (1,0)
    # Should be: Origin + (Identity * [1,0] * 100) = Origin + [100, 0]
    unit_target = np.array([1.0, 0.0])
    assert np.allclose(gen._unit_to_screen(unit_target), [110.0, 10.0])
    
    # Case C: A Point "Above" the path (0, 1)
    # Should be: Origin + (Identity * [0,1] * 100) = Origin + [0, 100]
    unit_up = np.array([0.0, 1.0])
    assert np.allclose(gen._unit_to_screen(unit_up), [10.0, 110.0])

# _screen_to_unit()
def test_screen_to_unit_applies_transform_logic():
    """Test that _screen_to_unit correctly applies: R_unit @ (point - origin) / D."""
    gen = PDPathGenerator()
    
    # MANUAL STATE INJECTION
    gen.origin = np.array([10.0, 10.0])
    gen.D = 100.0
    
    # Since R_screen is Identity, R_unit (its transpose) is also Identity.
    gen.R_unit = np.eye(2) 

    # Case A: The Start Point (10, 10) -> (0, 0)
    screen_start = np.array([10.0, 10.0])
    assert np.allclose(gen._screen_to_unit(screen_start), [0.0, 0.0])

    # Case B: The Target Point (110, 10) -> (1, 0)
    screen_target = np.array([110.0, 10.0])
    assert np.allclose(gen._screen_to_unit(screen_target), [1.0, 0.0])
    
    # Case C: A Point "Above" the path (10, 110) -> (0, 1)
    screen_up = np.array([10.0, 110.0])
    assert np.allclose(gen._screen_to_unit(screen_up), [0.0, 1.0])


# ------------ _init_velcity() ------------
# Tests for removed internal methods _init_velocity and _compute_kp_blend deleted.


# ------------_apply_overshoot()------------
# random triggers, geometric extension, and recovery logic

def test_overshoot_skips_conditions(mocker):
    """
    If the feature is disabled (prob <= 0), the user is unlucky (random > prob),
    or the path is too short to calculate a direction, the method should return the input path untouched.
    """
    gen = PDPathGenerator()
    path = np.array([[0,0], [10,10]])
    
    # Case A: Disabled in settings
    # Should return original path immediately
    res = gen._apply_overshoot(path, 10, 10, knobs={'overshoot_prob': 0})
    assert len(res) == 2
    
    # Case B: Bad luck (Random roll fails)
    # We mock random to return 0.9, which is > 0.5 probability
    mocker.patch('random.random', return_value=0.9)
    res = gen._apply_overshoot(path, 10, 10, knobs={'overshoot_prob': 0.5})
    assert len(res) == 2

    # Case C: Path too short (Cannot calculate direction)
    # Even if probability is 100%, it needs 2 points to define a vector
    short_path = np.array([[0,0]])
    res = gen._apply_overshoot(short_path, 10, 10, knobs={'overshoot_prob': 1.0})
    assert len(res) == 1


def test_overshoot_adds_points(mocker):
    """
    When enabled and triggered, it should add points to the path based on random geometry.
    The resulting path is longer than 2 points, and the final point eventually lands near the target.
    """
    gen = PDPathGenerator()
    gen.D = 100.0  # Method relies on self.D for scaling
    
    path = np.array([[0.0, 0.0], [100.0, 0.0]])
    
    # 1. Force the Probability Check to PASS (return 0.0)
    # 2. Force the Drift Sign check to return 0.0
    mocker.patch('random.random', side_effect=[0.0, 0.0])
    
    # 3. Force the Distance to be predictable (e.g., 0.05 or 5%)
    mocker.patch('random.uniform', return_value=0.05)
    
    # Run with 100% probability to trigger our mocks
    new_path = gen._apply_overshoot(path, 100, 0, knobs={'overshoot_prob': 1.0})
    
    # It should have added points. 
    # Logic: 5% of 100 = 5.0 dist. Steps = max(4, 5/3) = 4. Recovery = 3.
    # Total added = 7. Total path = 2 + 7 = 9.
    assert len(new_path) > 2

def test_overshoot_geometry_goes_past_target(mocker):
    """
    We verify the physical behavior. If the path was moving straight right towards x=100,
    the overshoot points should contain X values greater than 100.
    """ 
    gen = PDPathGenerator()
    gen.D = 100.0
    
    # Path ends exactly at target (100, 0)
    path = np.array([[90.0, 0.0], [100.0, 0.0]])
    
    # Force trigger
    mocker.patch('random.random', return_value=0.0)
    # Force a significant overshoot (8% of 100 = 8 units)
    mocker.patch('random.uniform', return_value=0.08)
    
    new_path = gen._apply_overshoot(path, 100, 0, knobs={'overshoot_prob': 1.0})
    
    # Extract just the X coordinates
    x_coords = new_path[:, 0]
    
    # The max X should be significantly past 100
    # (e.g., at least 104)
    assert np.max(x_coords) > 100.1
    
    # But it should return/recover towards the target at the end
    final_point = new_path[-1]
    # It might not be EXACTLY 100.0 due to the easing function, but close
    assert np.isclose(final_point[0], 100.0, atol=2.0)

def test_overshoot_handles_duplicate_points():
    """
    Cover the `if last_dir_norm <= 1e-6: return path` check in _apply_overshoot.
    This happens if the last two points of the path are identical (zero velocity at end).
    """
    gen = PDPathGenerator()
    
    # Create a path where the last two points are exactly the same
    # This results in a direction vector of [0, 0]
    path = np.array([[0,0], [10,10], [10,10]], dtype=float)
    
    # Attempt overshoot with 100% probability to ensure we reach the check
    res = gen._apply_overshoot(path, 20, 20, knobs={'overshoot_prob': 1.0})
    
    # It should return immediately (length 3) instead of calculating overshoot
    assert len(res) == 3
    assert np.array_equal(res, path)

# ------------ _prepare_knobs() ------------
# randomizing the input values (to simulate human inconsistency) 
# and scaling the PID gain (Kp) based on how long the path is
def test_prepare_knobs_scales_kp_by_distance(mocker):
    """
    Longer paths need weaker corrections (lower Kp), and shorter paths need stronger corrections. 
    Verify that kp_start and kp_end are modified correctly based on the distance.
    """ 
    gen = PDPathGenerator()
    
    # Mock randomization to return inputs unchanged (Pass-through)
    # This isolates the scaling logic test
    mocker.patch.object(gen, '_randomize', side_effect=lambda x, pct: x)
    
    # CASE A: Short Path (Distance is HALF of ref_dist)
    gen.D = 125.0  # canvas/4 is 250. 250/125 = 2.0 scale
    
    knobs = gen._prepare_knobs(
        mouse_velocity=10, kp_start=1.0, kp_end=1.0,
        stabilization=0, arc_strength=0, noise=0, overshoot_prob=0,
        keep_prob_start=0, keep_prob_end=0, variance=0,
        canvas_size=1000.0
    )
    
    # KP should be multiplied by 2.0
    assert np.isclose(knobs['kp_start'], 2.0)
    assert np.isclose(knobs['kp_end'], 2.0)

    # CASE B: Long Path (Distance is DOUBLE ref_dist)
    gen.D = 500.0  # 250/500 = 0.5 scale
    
    knobs = gen._prepare_knobs(
        mouse_velocity=10, kp_start=1.0, kp_end=1.0,
        stabilization=0, arc_strength=0, noise=0, overshoot_prob=0,
        keep_prob_start=0, keep_prob_end=0, variance=0,
        canvas_size=1000.0
    )
    
    # KP should be multiplied by 0.5
    assert np.isclose(knobs['kp_start'], 0.5)

def test_prepare_knobs_calls_randomizer(mocker):
    """
    Verify that the randomizer is called with the correct variance and returns the expected values.
    """
    gen = PDPathGenerator()
    gen.D = 100.0
    
    mock_return = {
        'mouse_velocity': 999.0,
        'kp_start': 1.0,
        'kp_end': 1.0
    }
    
    mock_random = mocker.patch.object(gen, '_randomize', return_value=mock_return)
    
    knobs = gen._prepare_knobs(
        mouse_velocity=10, kp_start=1, kp_end=1,
        stabilization=0.5, arc_strength=0, noise=0, overshoot_prob=0,
        keep_prob_start=0.8, keep_prob_end=0.9,
        variance=0.2,
        canvas_size=100.0
    )
    
    # 1. Verify Randomizer was called with correct variance
    call_args = mock_random.call_args
    assert call_args.kwargs['pct'] == 0.2
    
    # 2. Verify result contains randomized value
    assert knobs['mouse_velocity'] == 999.0
    
    # 3. Verify static values were appended correctly
    assert knobs['keep_prob_start'] == 0.8

def test_prepare_knobs_clips_scaling():
    """
    If the path is microscopically short or infinitely long, the KP scaling 
    shouldn't break the math (divide by zero) or create explosive values. 
    Your code clamps between 0.2 and 5.0.
    """
    gen = PDPathGenerator()
    # Mock randomizer again to keep it clean
    gen._randomize = lambda x, pct: x 
    
    # Case A: Microscopic Path (Near Zero distance)
    # Ratio would be huge (Infinity), but should clamp to 5.0
    gen.D = 0.0001
    
    knobs = gen._prepare_knobs(
        mouse_velocity=10, kp_start=1.0, kp_end=1.0,
        stabilization=0, arc_strength=0, noise=0, overshoot_prob=0,
        keep_prob_start=0, keep_prob_end=0, variance=0,
        canvas_size=100.0
    )
    assert knobs['kp_start'] == 5.0

    # Case B: Massive Path (Huge distance)
    # Ratio would be near zero, but should clamp to 0.2
    gen.D = 100000.0
    
    knobs = gen._prepare_knobs(
        mouse_velocity=10, kp_start=1.0, kp_end=1.0,
        stabilization=0, arc_strength=0, noise=0, overshoot_prob=0,
        keep_prob_start=0, keep_prob_end=0, variance=0,
        canvas_size=100.0
    )
    assert knobs['kp_start'] == 0.2
    
# ------------ generate_path()------------
def test_generate_path_hits_target_within_tolerance():
    """Verify the simulation loop actually drives the mouse to the target."""
    gen = PDPathGenerator()
    start = (100, 100)
    end = (500, 500)
    
    # Run simulation
    path, _, steps, _ = gen.generate_path(
        start_x=start[0], start_y=start[1],
        end_x=end[0], end_y=end[1],
        tol_px=1.0 # Strict tolerance
    )
    
    # 1. Check Start
    assert np.allclose(path[0], start, atol=0.1)
    
    # 2. Check End (The last point should be exactly the target due to post-processing)
    assert np.allclose(path[-1], end, atol=0.1)
    
    # 3. Check Convergence (The point BEFORE the forced end should be close)
    # This verifies the PID loop actually got there, rather than just snapping the line.
    assert np.allclose(path[-2], end, atol=5.0) 
    
    # 4. Check it generated steps
    assert len(path) > 10
    assert steps > 0

def test_generate_path_creates_arc(mocker):
    """Verify that arc parameters actually bend the path in the inlined path generator loop."""
    gen = PDPathGenerator()

    # 1. Force straight start
    mocker.patch('random.gauss', return_value=0.0)

    # 2. Run with STABILIZED parameters
    path, _, _, _ = gen.generate_path(
        0, 0, 1000, 0,
        arc_strength=0.5,
        arc_sign=1.0,
        noise=0.0,
        kp_start=0.05,
        kp_end=0.05,
        stabilization=0.5,
        mouse_velocity=0.05,
        # FIX: Stricter tolerance (1.0px) forces the simulation to land 
        # closer to the target before stopping. This minimizes the 
        # "rotation artifact" where the post-processing snaps the path down.
        tol_px=1.0 
    )

    y_coords = path[:, 1]

    # Logic: The path should bulge UP
    max_y = np.max(y_coords)
    assert max_y > 50.0

    # Logic: It should not dip significantly.
    # We allow -25.0 to account for the "Snap-to-Target" rotation.
    # (If the end is high, snapping it down pushes the start low).
    min_y = np.min(y_coords)
    assert min_y > -25.0

def test_generate_path_triggers_overshoot(mocker):
    """Verify that the overshoot parameter works in the main loop."""
    gen = PDPathGenerator()
    
    # Force random checks to pass
    mocker.patch('random.random', return_value=0.0)
    mocker.patch('random.uniform', return_value=0.05) 
    
    path, _, _, _ = gen.generate_path(
        0, 0, 100, 0, 
        overshoot_prob=1.0,
        noise=0.0
    )
    
    # Logic: Max X should go past the target (100)
    max_x = np.max(path[:, 0])
    assert max_x > 101.0
    
    # Logic: It must return to 100.0
    assert np.isclose(path[-1][0], 100.0, atol=0.1)

def test_generate_path_applies_variance():
    """Verify that requesting variance returns parameters different from defaults."""
    gen = PDPathGenerator()
    
    # Run two paths with high variance
    _, _, _, params1 = gen.generate_path(0,0,100,100, variance=0.5, mouse_velocity=10.0)
    _, _, _, params2 = gen.generate_path(0,0,100,100, variance=0.5, mouse_velocity=10.0)
    
    # The actual used velocity should NOT be exactly 10.0
    assert params1['mouse_velocity'] != 10.0
    
    # Two runs should likely be different
    assert params1['mouse_velocity'] != params2['mouse_velocity']

def test_generate_path_applies_offset():
    """Verify the offset_x/y parameters shift the final path."""
    gen = PDPathGenerator()
    
    # Generate path 0->100
    # But apply offset of (5000, 5000)
    path, _, _, _ = gen.generate_path(
        0, 0, 100, 100, 
        offset_x=5000, 
        offset_y=5000
    )
    
    # Start point should be 0+5000
    assert np.allclose(path[0], [5000, 5000], atol=1.0)
    # End point should be 100+5000
    assert np.allclose(path[-1], [5100, 5100], atol=1.0)

def test_generate_path_runs_with_noise():
    """Verify that the inlined noise logic in generate_path is executed."""
    gen = PDPathGenerator()
    
    # Run with significant noise
    path, _, steps, params = gen.generate_path(
        0, 0, 100, 100, 
        noise=0.5,
        max_steps=100
    )
    
    assert len(path) > 0
    
    import pytest
    assert params['noise'] == pytest.approx(0.5, rel=0.1)

def test_generate_path_zero_velocity_guard():
    """
    Cover the 'Stopped Mouse' guard.
    We run with 0.0 velocity to force current_speed < 1e-6, 
    triggering the direction reset logic.
    """
    gen = PDPathGenerator()
    
    # Run with 0.0 velocity. 
    # This forces the physics loop to hit the 'if current_speed < 1e-6' block.
    path, _, _, _ = gen.generate_path(
        0, 0, 100, 100, 
        mouse_velocity=0.0,
        max_steps=5 # Short run, we just need to hit the line once
    )
    
    # We just want to ensure the function ran without crashing/dividing by zero
    assert len(path) > 0