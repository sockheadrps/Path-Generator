from .generator import PDPathGenerator
try:
    from .pathemulator import PathEmulator
except ImportError:
    # Optional dependency missing
    PathEmulator = None

_path_generator = None

def get_path_generator() -> PDPathGenerator:
    """Get the global PD path generator instance."""
    global _path_generator
    if _path_generator is None:
        _path_generator = PDPathGenerator()
    return _path_generator

def generate_mouse_path(*args, **kwargs):
    """
    Convenience function to generate a mouse path.
    Returns (path_points, path_progress, total_steps, actual_params).
    """
    generator = get_path_generator()
    return generator.generate_path(*args, **kwargs)
