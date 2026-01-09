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
    2. Simulate movement with feedforward velocity and PD correction (inlined for performance)
    3. Apply human-like noise, stabilization, and velocity profiles
    4. Transform back to screen coordinates
    
    The physics simulation loop integrates several behaviors:
    
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
        'mouse_velocity', 'kp_start', 'kp_end', 'stabilization', 'noise',
        'keep_prob_start', 'keep_prob_end', 'arc_strength', 'variance',
        'overshoot_prob'
    )
    
    # Default values for preset parameters
    DEFAULT_PRESET = {
        'mouse_velocity': 0.65,
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
        
        # Validate keys
        supported = set(PDPathGenerator.PRESET_KEYS)
        unknown = set(data.keys()) - supported
        if unknown:
            raise ValueError(f"Unknown parameters in preset file: {unknown}. Supported: {supported}")

        # Only load recognized keys (safety reduntant but keeps logic clean)
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
            >>> jittered = PDPathGenerator._randomize({'x': 10.0, 'mouse_velocity': 0.35}, pct=0.1)
            >>> 9.0 <= jittered['x'] <= 11.0
            True
            >>> 0.315 <= jittered['mouse_velocity'] <= 0.385  # ±10% of 0.35
            True
        """
        out = {}
        for k, v in params.items():
            jitter = 1.0 + random.uniform(-pct, pct)
            out[k] = v * jitter
        return out

    # -------------------- Coordinate Transforms --------------------
    
    def _setup_transforms(self, start: np.ndarray, target: np.ndarray) -> None:
            # 1. Get the transform that aligns Screen -> Unit (flattens the line)
            R_align, self.D = get_unit_transform(start, target)
            
            # 2. Store it as R_unit
            self.R_unit = R_align
            
            # 3. The inverse (Transpose) rotates Unit -> Screen (lifts the line)
            self.R_screen = R_align.T
            
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
    
    # _init_velocity and _compute_kp_blend removed (logic inlined in generate_path)

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

        # ensure overshoot_points ends with target_pt    
        overshoot_points.append(target_pt)
        
        return np.vstack([path, overshoot_points])

    # -------------------- Knob Preparation --------------------

    def _prepare_knobs(
        self,
        mouse_velocity: float,
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
            mouse_velocity: Base velocity magnitude.
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
                mouse_velocity=mouse_velocity,
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
        mouse_velocity: Optional[float] = None,
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
            mouse_velocity: Base velocity (default: from preset).
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
            ...     mouse_velocity=0.35,
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
        
        # Resolve parameters from arguments or current preset
        p = self.preset
        mouse_velocity = mouse_velocity if mouse_velocity is not None else p['mouse_velocity']
        kp_start = kp_start if kp_start is not None else p['kp_start']
        kp_end = kp_end if kp_end is not None else p['kp_end']
        stabilization = stabilization if stabilization is not None else p['stabilization']
        noise = noise if noise is not None else p['noise']
        keep_prob_start = keep_prob_start if keep_prob_start is not None else p['keep_prob_start']
        keep_prob_end = keep_prob_end if keep_prob_end is not None else p['keep_prob_end']
        arc_strength = arc_strength if arc_strength is not None else p['arc_strength']
        variance = variance if variance is not None else p['variance']
        overshoot_prob = overshoot_prob if overshoot_prob is not None else p['overshoot_prob']
        
        canvas_size = max(canvas_width, canvas_height)
        
        max_px_step = np.clip(self.D / 30.0, 12.0, 50.0)

        max_step_units = max_px_step / max(self.D, 1e-6)

        knobs = self._prepare_knobs(
            mouse_velocity, kp_start, kp_end, stabilization, arc_strength,
            noise, overshoot_prob, keep_prob_start, keep_prob_end,
            variance, canvas_size
        )
        
        SPEED = knobs["mouse_velocity"]

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
        last_saved_px = start_screen

        # ---------------------------------------------------------
        # PERFORMANCE OPTIMIZATION: Inline Physics Loop
        # ---------------------------------------------------------
        # Python function calls are expensive. For maximum speed, 
        # we inline the physics logic directly into the main loop.
        # ---------------------------------------------------------

        # Pre-calculated constants
        MAX_STEP_M = max_step_units
        NOISE_STRENGTH = knobs["noise"]
        STABILIZATION = knobs["stabilization"]
        KP_START = knobs["kp_start"]
        KP_END = knobs["kp_end"]
        ARC_STR = knobs["arc_strength"]
        OVERSHOOT_PROB = knobs["overshoot_prob"]
        
        # Keep probability interpolation factors
        KP_S = knobs["keep_prob_start"]
        KP_E = knobs["keep_prob_end"]
        
        # Noise constants
        SIGMA = NOISE_STRENGTH * 0.002
        THETA = 0.15

        # PD constants
        KI = 0.0005
        PD_GAIN = 20.0

        # Run loop
        while steps < max_steps:
            # 1. Update State
            s = float(np.clip(P_unit[0], 0.0, 1.0))

            if steps == 0:
                # _init_velocity inline
                angle_err = random.gauss(0, 0.3)
                c, s_ang = np.cos(angle_err), np.sin(angle_err)
                rot = np.array([[c, -s_ang], [s_ang, c]], dtype=np.float32)
                vprev_unit = rot @ np.array([SPEED, 0.0], dtype=np.float32)
            
            # 2. Feedforward + Braking (Fitts's Law)
            current_speed = np.linalg.norm(vprev_unit)
            if current_speed < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = vprev_unit / current_speed
            
            dist_rem = 1.0 - s
            brake = float(np.clip(dist_rem * 4.0, 0.15, 1.0))
            v_unit = direction * (SPEED * brake)

            # 3. PD Correction
            # _compute_kp_blend inline
            if s < 0.5:
                weight_far = 1.0 - (s / 0.5)
                current_kp = weight_far * KP_START
            else:
                weight_near = (s - 0.5) / 0.5
                current_kp = weight_near * KP_END
            
            current_kp *= brake

            if current_kp > 1e-6:
                # _compute_arc_offset inline
                if ARC_STR > 1e-4:
                    ideal_y = arc_sign * ARC_STR * np.sin(s * np.pi)
                else:
                    ideal_y = 0.0
                
                err_x = 1.0 - P_unit[0]
                err_y = ideal_y - P_unit[1]
                
                # Manual array creation is faster than np.array usually, 
                # but we need vector math.
                # err_unit = np.array([err_x, err_y], np.float32)
                
                err_sum_unit[0] += err_x
                err_sum_unit[1] += err_y
                
                # v_unit = v_unit + current_kp * 20.0 * err_unit + ki * err_sum_unit
                # Unrolled for speed:
                v_unit[0] += current_kp * PD_GAIN * err_x + KI * err_sum_unit[0]
                v_unit[1] += current_kp * PD_GAIN * err_y + KI * err_sum_unit[1]

            # 4. Correlated Noise (Ornstein-Uhlenbeck)
            if NOISE_STRENGTH > 1e-4:
                nf = (1.0 - s) ** 1.3
                
                # Random kick
                r1 = random.gauss(0, 1)
                r2 = random.gauss(0, 1)
                
                # Update noise state: noise_state = noise_state - theta*noise_state + sigma*kick
                ns_x = noise_state[0]
                ns_y = noise_state[1]
                
                noise_state[0] = ns_x + (-THETA * ns_x) + (SIGMA * r1)
                noise_state[1] = ns_y + (-THETA * ns_y) + (SIGMA * r2)
                
                v_unit[0] += noise_state[0] * nf
                v_unit[1] += noise_state[1] * nf

            # 5. Stabilization
            damp_val = STABILIZATION * 0.5
            if damp_val > 1e-6:
                # v_unit = v_unit - damp_val * (v_unit - vprev_unit)
                v_unit[0] -= damp_val * (v_unit[0] - vprev_unit[0])
                v_unit[1] -= damp_val * (v_unit[1] - vprev_unit[1])
            
            alpha = max(0.01, 1.0 - STABILIZATION)
            # v_unit = alpha * v_unit + (1.0 - alpha) * vprev_unit
            v_unit[0] = alpha * v_unit[0] + (1.0 - alpha) * vprev_unit[0]
            v_unit[1] = alpha * v_unit[1] + (1.0 - alpha) * vprev_unit[1]

            # 6. Limit Step
            mag = float(np.hypot(v_unit[0], v_unit[1])) + 1e-8
            step_limit = MAX_STEP_M * (0.5 + 0.5 * (1.0 - s) ** 1.5)
            if mag > step_limit:
                scale = step_limit / mag
                v_unit[0] *= scale
                v_unit[1] *= scale
            
            max_forward = max(0.0, 1.0 - P_unit[0])
            if v_unit[0] > max_forward:
                v_unit[0] = max_forward

            # 7. Integrate
            # P_next = P_unit + v_unit
            pn_x = P_unit[0] + v_unit[0]
            pn_y = P_unit[1] + v_unit[1]
            
            # Clip X
            if pn_x > 1.0: pn_x = 1.0
            elif pn_x < -0.05: pn_x = -0.05
            
            # Calculate screen pos manually to avoid function call overhead
            # P_screen = origin + (Pu * D) @ R_screen
            # This matrix mul is unavoidable but expensive.
            # R_screen is 2x2.
            # Let's perform the transform inline.
            # P_unit_scaled = (pn_x * D, pn_y * D)
            pus_x = pn_x * self.D
            pus_y = pn_y * self.D
            
            # rot is (2,2)
            # res = [pus_x*R00 + pus_y*R10, pus_x*R01 + pus_y*R11] + origin
            
            new_px_x = self.origin[0] + (pus_x * self.R_screen[0,0] + pus_y * self.R_screen[1,0])
            new_px_y = self.origin[1] + (pus_x * self.R_screen[0,1] + pus_y * self.R_screen[1,1])
            
            # Update P_unit (for next iteration, usually just P_next unless we clamped/transformed back)
            # Since we just clipped pn_x, we can use that directly for the loop state
            P_unit[0] = pn_x
            P_unit[1] = pn_y
            
            vprev_unit[0] = v_unit[0]
            vprev_unit[1] = v_unit[1]

            # 8. Store Point
            keep_p = KP_S + (KP_E - KP_S) * s
            
            should_keep = random.random() < keep_p or s >= 0.97
            
            if should_keep:
                path.append((float(new_px_x), float(new_px_y)))
                prog.append(s)
                last_saved_px[0] = new_px_x
                last_saved_px[1] = new_px_y

            # Check termination
            # dist_target = hypot(target - new)
            dtx = target_px[0] - new_px_x
            dty = target_px[1] - new_px_y
            if (dtx*dtx + dty*dty) <= (tol_px * tol_px) or s >= 0.995:
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