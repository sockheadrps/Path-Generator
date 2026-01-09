
import time
import sys
from typing import List, Tuple, Callable, Optional

# try to import win32api, else warn or fallback?
# The user explicitly asked for win32api, so we assume it defines the requirement.
try:
    import win32api
except ImportError:
    win32api = None


class PathEmulator:
    """
    Handles mouse cursor emulation using win32api for high performance.
    Requires 'pywin32' to be installed.
    """
    def __init__(self):
        if win32api is None:
            raise ImportError("win32api is required for PathEmulator. Please install 'pywin32'.")

    def get_position(self) -> Tuple[int, int]:
        """Get current cursor position (x, y) in screen coordinates."""
        return win32api.GetCursorPos()

    def execute_path(
        self,
        path: List[Tuple[float, float]],
        delay_between_points: float = 0.01
    ) -> None:
        """
        Replays a sequence of (x, y) coordinates by moving the mouse cursor.
        
        Args:
            path: List of (x, y) tuples representing the path.
            delay_between_points: Time in seconds to sleep between points. 
                                  Use 0.0 for maximum speed.
        """
        for x, y in path:
            win32api.SetCursorPos((int(x), int(y)))
            
            if delay_between_points > 0:
                time.sleep(delay_between_points)
