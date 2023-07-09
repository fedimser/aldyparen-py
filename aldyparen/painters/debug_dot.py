import numpy as np


class DebugDotPainter:
    def __init__(self, x=1.0, y=0.5, radius=0.1):
        self.x = x
        self.y = y
        self.radius = radius
        self.center = np.complex128(x + 1j * y)

    def to_object(self) -> object:
        return {
            "x": self.x,
            "y": self.y,
            "radius": self.radius
        }

    def paint(self, points: 'np.ndarray', ans: 'np.ndarray') -> 'np.ndarray':
        p = np.abs(self.center - points) < self.radius
        ans[:] = np.array(p, dtype=np.uint32)
