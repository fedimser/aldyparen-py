import warnings
from typing import Any, Callable

import numba
import numpy as np
from numpy.typing import NDArray

from ..util import prepare_function
from .base import Painter


class MandelbroidPainter(Painter):
    def __init__(self, gen_function: str = "z*z+c", max_iter: int = 100, radius: float = 2):
        assert 1 <= max_iter <= 1000000, "bad max_iter"
        self.gen_function = gen_function
        self.max_iter = max_iter
        self.radius = radius
        self.gen_function_prepared = prepare_function(gen_function, variables=["c", "z"])
        self.paint_func: Callable[[np.ndarray], np.ndarray] | None = None

    def to_object(self) -> dict[str, Any]:
        return {"gen_function": self.gen_function, "radius": self.radius, "max_iter": self.max_iter}

    def paint(self, points: NDArray[np.complex128], ans: NDArray[np.uint32]) -> None:
        if self.paint_func is None:
            numba_namespace: dict[str, Any] = {"numba": numba, "np": np}
            source = "\n".join(
                [
                    f'@numba.vectorize("u4(c16)", target="parallel")',
                    f"def painter__(c):",
                    f"  z = 0",
                    f"  for i in range({self.max_iter}):",
                    f"    z = {self.gen_function_prepared}",
                    f"    if np.abs(z) > {self.radius}: return i",
                    f"  return {self.max_iter}",
                ]
            )
            exec(source, numba_namespace)
            self.paint_func = numba_namespace["painter__"]

        assert self.paint_func is not None
        warnings.filterwarnings("ignore", message="overflow")
        try:
            ans[:] = self.paint_func(points)
        except Exception as e:
            self.warning = f"Error in mandelbroid painter: {e}"
            ans[:] = np.zeros_like(points, dtype=np.uint32)
