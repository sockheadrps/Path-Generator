"""PD-driven mouse path generator with modular transformation steps.

This module provides the main path generation class that creates human-like
mouse movement trajectories using a PD (Proportional-Derivative) control
approach combined with various transformations for realism.

Example:
    Basic usage::
    
        from pathgenerator import PDPathGenerator
        
        gen = PDPathGenerator()
        path, progress, steps, params = gen.generate_path(
            start_x=100, start_y=200,
            end_x=500, end_y=400,
            speed=0.35,
            noise=0.2
        )
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import numpy as np

from .geometry import get_unit_transform, rotate_scale_path_to_hit_target


class PDPathGenerator:
    """Human-like mouse path generator using PD control.
    
    Uses a standard unit-frame approach for resolution-independent path generation:
    
    1. Transform the problem so start=(0,0) and target=(1,0)
    2. Simulate movement with feedforward velocity and PD correction
    3. Apply human-like noise, stabilization, and velocity profiles
    4. Transform back to screen coordinates
    
    Each transformation step is implemented as a separate method for testability
    and extensibility. The transformation pipeline per simulation step is:
    
    1. Feedforward + braking (Fitts's Law deceleration)
    2. PD correction (steer toward target/arc)
    3. Correlated noise (hand tremor)
    4. Stabilization (damping + smoothing)
    5. Step limiting (prevent jumps and backtracking)
    
    Attributes:
        R_screen: Rotation matrix for unit→screen conversion.
        R_unit: Rotation matrix for screen→unit conversion (R_screen.T).
        D: Distance between start and target in screen pixels.
        origin: Start point in screen coordinates.
    
    Example:
        >>> gen = PDPathGenerator()
        >>> path, progress, steps, params = gen.generate_path(
        ...     start_x=100, start_y=200,
        ...     end_x=500, end_y=400,
        ...     speed=0.35,
        ...     noise=0.2,
        ...     arc_strength=0.15
        ... )
        >>> len(path)  # Number of points in path
        47
    """

    # Keys that are loaded from preset files
    PRESET_KEYS = (
        'speed', 'kp_start', 'kp_end', 'stabilization', 'noise',
        'keep_prob_start', 'keep_prob_end', 'arc_strength', 'variance',
        'overshoot_prob'
    )
    
    # Default values for preset parameters
    DEFAULT_PRESET = {
        'speed': 0.35,
        'kp_start': 0.01,
        'kp_end': 0.01,
        'stabilization': 0.15,
        'noise': 0.0,
        'keep_prob_start': 0.70,
        'keep_prob_end': 0.98,
        'arc_strength': 0.0,
        'variance': 0.0,
        'overshoot_prob': 0.0
    }

    def __init__(self, preset_file: Optional[str] = None):
        """Initialize the path generator.
        
        Args:
            preset_file: Optional path to a JSON file with preset parameters.
                         If provided, these values become the defaults for generate_path().
        
        Example:
            >>> gen = PDPathGenerator('natural.json')
            >>> path, *_ = gen.generate_path(100, 200, 500, 400)  # Uses preset values
        """
        self.R_screen: Optional[np.ndarray] = None
        self.R_unit: Optional[np.ndarray] = None
        self.D: float = 0.0
        self.origin: Optional[np.ndarray] = None
        
        # Load preset or use defaults
        if preset_file is not None:
            self.preset = self._load_preset_file(preset_file)
        else:
            self.preset = self.DEFAULT_PRESET.copy()
    
    @staticmethod
    def _load_preset_file(filepath: str) -> dict:
        """Load parameter preset from a JSON file.
        
        Args:
            filepath: Path to the JSON file.
        
        Returns:
            Dictionary of parameter values merged with defaults.
        """
        import json
        
        preset = PDPathGenerator.DEFAULT_PRESET.copy()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Only load recognized keys
        for key in PDPathGenerator.PRESET_KEYS:
            if key in data:
                preset[key] = data[key]
        
        return preset

    # -------------------- Parameter Randomization --------------------
    
    @staticmethod
    def _randomize(params: dict, pct: float = 0.10) -> dict:
        """Apply random jitter to parameter values.
        
        Args:
            params: Dictionary of parameter names to values.
            pct: Maximum percentage variation (0.10 = ±10%).
        
        Returns:
            New dictionary with jittered values.
        
        Example:
            >>> params = {'speed': 0.35, 'noise': 0.2}
            >>> jittered = PDPathGenerator._randomize(params, pct=0.1)
            >>> 0.315 <= jittered['speed'] <= 0.385  # ±10% of 0.35
            True
        """
        out = {}
        for k, v in params.items():
            jitter = 1.0 + random.uniform(-pct, pct)
            out[k] = v * jitter
        return out

    # -------------------- Coordinate Transforms --------------------
    
    def _setup_transforms(self, start: np.ndarray, target: np.ndarray) -> None:
        """Initialize unit-frame transforms.
        
        Sets up rotation matrices and distance for converting between
        screen coordinates and the normalized unit frame where:
        
        - start = (0, 0)
        - target = (1, 0)
        
        Args:
            start: Starting point as numpy array [x, y].
            target: Target point as numpy array [x, y].
        """
        self.R_screen, self.D = get_unit_transform(start, target)
        self.R_unit = self.R_screen.T
        self.origin = start

    def _unit_to_screen(self, Pu: np.ndarray) -> np.ndarray:
        """Convert unit-frame point to screen coordinates.
        
        Args:
            Pu: Point in unit frame as [x, y] array.
        
        Returns:
            Point in screen coordinates as float32 array.
        """
        return (self.origin + (Pu * self.D) @ self.R_screen).astype(np.float32)

    def _screen_to_unit(self, Ps: np.ndarray) -> np.ndarray:
        """Convert screen point to unit-frame.
        
        Args:
            Ps: Point in screen coordinates as [x, y] array.
        
        Returns:
            Point in unit frame as float32 array.
        """
        return ((Ps - self.origin) @ self.R_unit / max(self.D, 1e-6)).astype(np.float32)

    # -------------------- Velocity Transformations --------------------
    
    def _init_velocity(self, speed: float) -> np.ndarray:
        """Create initial velocity with random angle error.
        
        Simulates human imprecision at movement start. The initial direction
        has a random angular error with ~17 degrees standard deviation.
        
        Args:
            speed: Base velocity magnitude in unit space.
        
        Returns:
            Initial velocity vector as float32 array.
        """
        angle_err = random.gauss(0, 0.3)
        c, s_ang = np.cos(angle_err), np.sin(angle_err)
        rot = np.array([[c, -s_ang], [s_ang, c]], dtype=np.float32)
        return (rot @ np.array([speed, 0.0], dtype=np.float32))

    def _apply_feedforward(
        self, 
        direction: np.ndarray, 
        speed: float, 
        progress: float
    ) -> Tuple[np.ndarray, float]:
        """Apply feedforward velocity with Fitts's Law style braking.
        
        Creates the characteristic velocity profile where movement is fast
        in the middle and slow near the target. The braking follows Fitts's
        Law: humans naturally slow down as they approach a target.
        
        Args:
            direction: Unit vector of current movement direction.
            speed: Base velocity magnitude in unit space.
            progress: Path completion ratio (0.0 at start, 1.0 at target).
        
        Returns:
            Tuple containing:
                - velocity: The feedforward velocity vector.
                - brake: The braking factor (0.15 to 1.0).
        """
        dist_rem = 1.0 - progress
        brake = float(np.clip(dist_rem * 4.0, 0.15, 1.0))
        v_unit = direction * (speed * brake)
        return v_unit, brake

    def _compute_kp_blend(self, progress: float, knobs: dict) -> float:
        """Compute blended KP value based on progress along path.
        
        Uses distance-weighted blending between kp_start and kp_end:
        
        - First half (0-50%): kp_start dominates, fading to 0 at midpoint
        - Second half (50-100%): kp_end grows from 0 to full strength
        
        This allows different correction behavior at the start (coarse
        acquisition) vs. end (fine targeting).
        
        Args:
            progress: Path completion ratio (0.0 to 1.0).
            knobs: Parameter dictionary containing 'kp_start' and 'kp_end'.
        
        Returns:
            Blended KP value for current progress.
        """
        if progress < 0.5:
            weight_far = 1.0 - (progress / 0.5)
            weight_near = 0.0
        else:
            weight_far = 0.0
            weight_near = (progress - 0.5) / 0.5
        
        return weight_far * knobs["kp_start"] + weight_near * knobs["kp_end"]

    def _compute_arc_offset(
        self, 
        progress: float, 
        arc_strength: float, 
        arc_sign: float
    ) -> float:
        """Calculate ideal Y offset for arc trajectory.
        
        Uses a sine wave to create a smooth arc that bulges in the middle
        and returns to the straight line at both endpoints:
        `y = arc_sign * arc_strength * sin(π * progress)`
        
        Args:
            progress: Path completion ratio (0.0 to 1.0).
            arc_strength: Maximum arc height (0.0 to ~0.5 typical).
            arc_sign: Direction of arc (+1 or -1).
        
        Returns:
            Ideal Y position in unit frame for smooth arc trajectory.
        """
        if arc_strength > 1e-4:
            return arc_sign * arc_strength * np.sin(progress * np.pi)
        return 0.0

    def _apply_pd_correction(
        self,
        v_unit: np.ndarray,
        P_unit: np.ndarray,
        progress: float,
        brake: float,
        knobs: dict,
        arc_sign: float,
        err_sum_unit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adaptive PD correction with arc offset.
        
        Steers the path toward the target (or arc trajectory) using
        proportional-integral control. The correction strength adapts
        based on progress and braking.
        
        Args:
            v_unit: Current velocity vector in unit frame.
            P_unit: Current position in unit frame.
            progress: Path completion ratio (0.0 to 1.0).
            brake: Current braking factor from feedforward.
            knobs: Parameter dictionary with 'kp_start', 'kp_end', 'arc_strength'.
            arc_sign: Direction of arc curve (+1 or -1).
            err_sum_unit: Accumulated error for integral term.
        
        Returns:
            Tuple containing:
                - v_unit: Modified velocity vector.
                - err_sum_unit: Updated accumulated error.
        """
        current_kp = self._compute_kp_blend(progress, knobs)
        current_kp *= brake

        if current_kp > 1e-6:
            ideal_y = self._compute_arc_offset(progress, knobs["arc_strength"], arc_sign)
            
            err_x = 1.0 - P_unit[0]
            err_y = ideal_y - P_unit[1]
            err_unit = np.array([err_x, err_y], np.float32)
            
            err_sum_unit = err_sum_unit + err_unit
            
            ki = 0.0005
            v_unit = v_unit + current_kp * 20.0 * err_unit + ki * err_sum_unit

        return v_unit, err_sum_unit

    def _apply_noise(
        self,
        v_unit: np.ndarray,
        noise_state: np.ndarray,
        progress: float,
        noise_strength: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Ornstein-Uhlenbeck correlated noise.
        
        Creates smooth, correlated random perturbations that simulate
        natural hand tremor and micro-imprecision. Unlike white noise,
        this produces smooth wandering motion.
        
        The noise automatically decays near the target to ensure accurate
        endpoint arrival.
        
        Args:
            v_unit: Current velocity vector in unit frame.
            noise_state: Current noise state for Ornstein-Uhlenbeck process.
            progress: Path completion ratio (0.0 to 1.0).
            noise_strength: Noise intensity (0.0 to 1.0 typical).
        
        Returns:
            Tuple containing:
                - v_unit: Modified velocity vector.
                - noise_state: Updated noise state.
        """
        if noise_strength <= 1e-4:
            return v_unit, noise_state
            
        theta = 0.15
        sigma = noise_strength * 0.002
        
        nf = (1.0 - progress) ** 1.3
        
        rand_kick = np.array([random.gauss(0, 1), random.gauss(0, 1)], dtype=np.float32)
        noise_state = noise_state + (-theta * noise_state) + (sigma * rand_kick)
        
        v_unit = v_unit + noise_state * nf
        
        return v_unit, noise_state

    def _apply_stabilization(
        self,
        v_unit: np.ndarray,
        vprev_unit: np.ndarray,
        stabilization: float
    ) -> np.ndarray:
        """Apply damping and smoothing to velocity.
        
        Combines two effects for natural motion:
        
        1. **Damping**: Reduces sudden velocity changes
        2. **Smoothing**: Blends current velocity with previous
        
        Higher stabilization creates smoother, more flowing paths but
        reduces responsiveness to corrections.
        
        Args:
            v_unit: Current velocity vector.
            vprev_unit: Previous velocity vector.
            stabilization: Stabilization factor (0.0 to 1.0).
        
        Returns:
            Stabilized velocity vector.
        """
        damp_val = stabilization * 0.5
        if damp_val > 1e-6:
            v_unit = v_unit - damp_val * (v_unit - vprev_unit)
        
        alpha = max(0.01, 1.0 - stabilization)
        v_unit = alpha * v_unit + (1.0 - alpha) * vprev_unit
        
        return v_unit

    def _limit_step(
        self,
        v_unit: np.ndarray,
        P_unit: np.ndarray,
        progress: float,
        max_step_units: float
    ) -> np.ndarray:
        """Limit step magnitude and prevent backing up.
        
        Ensures movement stays within reasonable bounds:
        
        1. Limits maximum step size (tighter near target)
        2. Prevents significant backward motion (X shouldn't decrease)
        
        Args:
            v_unit: Current velocity vector.
            P_unit: Current position in unit frame.
            progress: Path completion ratio (0.0 to 1.0).
            max_step_units: Maximum step size in unit space.
        
        Returns:
            Limited velocity vector.
        """
        mag = float(np.linalg.norm(v_unit)) + 1e-8
        step_limit = max_step_units * (0.5 + 0.5 * (1.0 - progress) ** 1.5)
        if mag > step_limit:
            v_unit = v_unit * (step_limit / mag)
        
        max_forward = max(0.0, 1.0 - P_unit[0])
        if v_unit[0] > max_forward:
            v_unit[0] = max_forward
        
        return v_unit

    def _integrate_step(
        self,
        P_unit: np.ndarray,
        v_unit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Integrate velocity to get next position.
        
        Performs Euler integration and converts through screen coordinates
        for numerical consistency.
        
        Args:
            P_unit: Current position in unit frame.
            v_unit: Current velocity in unit frame.
        
        Returns:
            Tuple containing:
                - new_px: New position in screen coordinates.
                - P_unit: New position in unit frame.
                - progress: New progress value (0.0 to 1.0).
        """
        P_next = (P_unit + v_unit).astype(np.float32)
        P_next[0] = float(np.clip(P_next[0], -0.05, 1.0))
        
        new_px = self._unit_to_screen(P_next)
        P_unit = self._screen_to_unit(new_px)
        progress = float(np.clip(P_unit[0], 0.0, 1.0))
        
        return new_px, P_unit, progress

    def _should_keep_point(self, progress: float, knobs: dict) -> bool:
        """Determine if current point should be kept based on density settings.
        
        Uses linear interpolation between start and end keep probabilities.
        Points near the end (progress >= 0.97) are always kept for accuracy.
        
        Args:
            progress: Path completion ratio (0.0 to 1.0).
            knobs: Parameter dictionary with 'keep_prob_start' and 'keep_prob_end'.
        
        Returns:
            True if the point should be included in the final path.
        """
        kp_s = knobs["keep_prob_start"]
        kp_e = knobs["keep_prob_end"]
        keep_p = kp_s + (kp_e - kp_s) * progress
        
        return random.random() < keep_p or progress >= 0.97

    # -------------------- Path Post-Processing --------------------

    def _apply_overshoot(
        self,
        path: np.ndarray,
        end_x: float,
        end_y: float,
        knobs: dict
    ) -> np.ndarray:
        """Add overshoot and recovery in screen space.
        
        Simulates the human tendency to overshoot targets and correct.
        When triggered, adds points that:
        
        1. Continue past the target (3-8% of path distance)
        2. Curve slightly perpendicular to add realism
        3. Decelerate and recover back to the target
        
        Args:
            path: Current path as (N, 2) numpy array.
            end_x: Target X coordinate.
            end_y: Target Y coordinate.
            knobs: Parameter dictionary with 'overshoot_prob'.
        
        Returns:
            Path with overshoot points appended (if triggered).
        """
        if knobs["overshoot_prob"] <= 0:
            return path
            
        if random.random() >= knobs["overshoot_prob"]:
            return path
            
        if len(path) < 2:
            return path
            
        last_dir = path[-1] - path[-2]
        last_dir_norm = np.linalg.norm(last_dir)
        
        if last_dir_norm <= 1e-6:
            return path
            
        last_dir = last_dir / last_dir_norm
        
        perp = np.array([-last_dir[1], last_dir[0]], dtype=np.float32)
        drift_sign = 1.0 if random.random() > 0.5 else -1.0
        
        overshoot_dist = self.D * random.uniform(0.03, 0.08)
        overshoot_steps = max(4, min(int(overshoot_dist / 3), 10))
        
        overshoot_points = []
        pos = path[-1].copy()
        
        for i in range(overshoot_steps):
            decel = 1.0 - (i / overshoot_steps) * 0.9
            step_size = (overshoot_dist / overshoot_steps) * decel
            curve = np.sin((i / overshoot_steps) * np.pi) * 1.5 * drift_sign
            
            pos = pos + last_dir * step_size + perp * curve
            overshoot_points.append(pos.copy())
        
        target_pt = np.array([end_x, end_y], dtype=np.float32)
        recovery_steps = max(3, overshoot_steps // 2)
        for i in range(recovery_steps):
            t = (i + 1) / recovery_steps
            ease = t * t * (3 - 2 * t)
            to_target = target_pt - pos
            pos = pos + to_target * ease * 0.5
            overshoot_points.append(pos.copy())
        
        return np.vstack([path, overshoot_points])

    # -------------------- Knob Preparation --------------------

    def _prepare_knobs(
        self,
        speed: float,
        kp_start: float,
        kp_end: float,
        stabilization: float,
        arc_strength: float,
        noise: float,
        overshoot_prob: float,
        keep_prob_start: float,
        keep_prob_end: float,
        variance: float,
        canvas_size: float
    ) -> dict:
        """Prepare and randomize control knobs for path generation.
        
        Applies variance-based randomization to most parameters and
        scales KP values based on path distance.
        
        Args:
            speed: Base velocity magnitude.
            kp_start: Correction strength at path start.
            kp_end: Correction strength near target.
            stabilization: Smoothing/damping factor.
            arc_strength: Curvature of arc trajectory.
            noise: Hand tremor intensity.
            overshoot_prob: Probability of overshooting target.
            keep_prob_start: Point density at path start.
            keep_prob_end: Point density at path end.
            variance: Random variation percentage for parameters.
            canvas_size: Larger canvas dimension for KP scaling.
        
        Returns:
            Dictionary of prepared parameter values.
        """
        knobs = self._randomize(
            dict(
                speed=speed,
                kp_start=kp_start,
                kp_end=kp_end,
                stabilization=stabilization,
                arc_strength=arc_strength,
                noise=noise,
                overshoot_prob=overshoot_prob
            ),
            pct=max(0.01, variance)
        )

        knobs['keep_prob_start'] = keep_prob_start
        knobs['keep_prob_end'] = keep_prob_end
        
        ref_dist = canvas_size / 4.0
        kp_scale = np.clip(ref_dist / (self.D + 1e-6), 0.2, 5.0)
        knobs["kp_start"] *= kp_scale
        knobs["kp_end"] *= kp_scale
        
        return knobs

    # -------------------- Main Generation --------------------

    def generate_path(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        speed: Optional[float] = None,
        kp_start: Optional[float] = None,
        kp_end: Optional[float] = None,
        stabilization: Optional[float] = None,
        noise: Optional[float] = None,
        keep_prob_start: Optional[float] = None,
        keep_prob_end: Optional[float] = None,
        arc_strength: Optional[float] = None,
        variance: Optional[float] = None,
        arc_sign: Optional[float] = None,
        overshoot_prob: Optional[float] = None,
        max_steps: int = 1200,
        tol_px: float = 3.0,
    ) -> Tuple[np.ndarray, List[float], int, Dict]:
        """Generate a human-like mouse path from start to end.
        
        This is the main entry point for path generation. It simulates
        human mouse movement using PD control with various transformations
        for realism.
        
        Args:
            start_x: Starting X coordinate in screen pixels.
            start_y: Starting Y coordinate in screen pixels.
            end_x: Target X coordinate in screen pixels.
            end_y: Target Y coordinate in screen pixels.
            canvas_width: Canvas/viewport width in pixels.
            canvas_height: Canvas/viewport height in pixels.
            offset_x: X offset to add to output coordinates (for window positioning).
            offset_y: Y offset to add to output coordinates (for window positioning).
            speed: Base velocity (default: from preset).
            kp_start: Correction strength at start (default: from preset).
            kp_end: Correction strength near target (default: from preset).
            stabilization: Smoothing factor (default: from preset).
            noise: Hand tremor intensity (default: from preset).
            keep_prob_start: Point density at start (default: from preset).
            keep_prob_end: Point density at end (default: from preset).
            arc_strength: Curvature amount (default: from preset).
            variance: Random parameter variation (default: from preset).
            arc_sign: Arc direction (+1 or -1), None for random.
            overshoot_prob: Overshoot probability (default: from preset).
            max_steps: Maximum simulation steps before termination.
            tol_px: Distance tolerance for target arrival (pixels).
        
        Returns:
            Tuple containing:
                - path: (N, 2) numpy array of screen coordinates.
                - progress: List of progress values (0-1) per kept point.
                - steps: Number of simulation steps executed.
                - params: Dictionary of actual parameters used (after randomization).
        
        Example:
            >>> gen = PDPathGenerator()
            >>> path, progress, steps, params = gen.generate_path(
            ...     start_x=100, start_y=200,
            ...     end_x=500, end_y=400,
            ...     speed=0.35,
            ...     noise=0.2,
            ...     arc_strength=0.15
            ... )
            >>> path.shape
            (47, 2)
            >>> path[0]  # Start point
            array([100., 200.], dtype=float32)
            >>> path[-1]  # End point (exact target)
            array([500., 400.], dtype=float32)
        """
        start_px = (float(start_x), float(start_y))
        target_px = (float(end_x), float(end_y))
        start_vec = np.array(start_px, np.float32)
        target_vec = np.array(target_px, np.float32)
        
        self._setup_transforms(start_vec, target_vec)
        
        canvas_size = max(canvas_width, canvas_height)
        max_px_step = canvas_size / 160.0
        max_step_units = max_px_step / max(self.D, 1e-6)

        # Resolve parameters from arguments or current preset
        p = self.preset
        speed = speed if speed is not None else p['speed']
        kp_start = kp_start if kp_start is not None else p['kp_start']
        kp_end = kp_end if kp_end is not None else p['kp_end']
        stabilization = stabilization if stabilization is not None else p['stabilization']
        noise = noise if noise is not None else p['noise']
        keep_prob_start = keep_prob_start if keep_prob_start is not None else p['keep_prob_start']
        keep_prob_end = keep_prob_end if keep_prob_end is not None else p['keep_prob_end']
        arc_strength = arc_strength if arc_strength is not None else p['arc_strength']
        variance = variance if variance is not None else p['variance']
        overshoot_prob = overshoot_prob if overshoot_prob is not None else p['overshoot_prob']

        knobs = self._prepare_knobs(
            speed, kp_start, kp_end, stabilization, arc_strength,
            noise, overshoot_prob, keep_prob_start, keep_prob_end,
            variance, canvas_size
        )
        
        SPEED = knobs["speed"]

        P_unit = self._screen_to_unit(np.array(start_px, np.float32))
        vprev_unit = np.array([0.0, 0.0], np.float32)
        err_sum_unit = np.array([0.0, 0.0], np.float32)
        noise_state = np.array([0.0, 0.0], dtype=np.float32)
        
        if arc_sign is None:
            arc_sign = 1.0 if random.random() > 0.5 else -1.0
        else:
            arc_sign = float(arc_sign)

        path: List[Tuple[float, float]] = []
        prog: List[float] = []
        start_screen = self._unit_to_screen(P_unit)
        path.append((float(start_screen[0]), float(start_screen[1])))
        prog.append(0.0)

        steps = 0
        while steps < max_steps:
            s = float(np.clip(P_unit[0], 0.0, 1.0))

            if steps == 0:
                vprev_unit = self._init_velocity(SPEED)
            
            current_speed = np.linalg.norm(vprev_unit)
            if current_speed < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = vprev_unit / current_speed
            
            v_unit, brake = self._apply_feedforward(direction, SPEED, s)

            v_unit, err_sum_unit = self._apply_pd_correction(
                v_unit, P_unit, s, brake, knobs, arc_sign, err_sum_unit
            )

            v_unit, noise_state = self._apply_noise(
                v_unit, noise_state, s, knobs["noise"]
            )

            v_unit = self._apply_stabilization(v_unit, vprev_unit, knobs["stabilization"])

            v_unit = self._limit_step(v_unit, P_unit, s, max_step_units)

            new_px, P_unit, s = self._integrate_step(P_unit, v_unit)
            vprev_unit = v_unit.copy()

            if self._should_keep_point(s, knobs):
                path.append((float(new_px[0]), float(new_px[1])))
                prog.append(s)

            dist_px = float(np.hypot(target_px[0] - new_px[0], target_px[1] - new_px[1]))
            if (dist_px <= tol_px) or s >= 0.995:
                break

            steps += 1

        path_array = np.array(path, dtype=np.float32)
        
        path_fixed = rotate_scale_path_to_hit_target(
            path_array, (start_x, start_y), (end_x, end_y), scale_to_distance=True
        )
        
        path_fixed = self._apply_overshoot(path_fixed, end_x, end_y, knobs)
        
        # Apply viewport offset if specified
        if offset_x != 0.0 or offset_y != 0.0:
            path_fixed = path_fixed + np.array([offset_x, offset_y], dtype=np.float32)
        
        return path_fixed, prog, steps, knobs
