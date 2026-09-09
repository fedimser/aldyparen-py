from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Painter(ABC):
    """Base class for ordinary-precision painters.

    Subclass constructors must accept keyword arguments and support construction
    with no arguments. Concrete subclasses implement :meth:`paint`, which
    receives complex128 coordinates. Painters that require digit-array
    coordinates must inherit :class:`HighPrecisionPainter` instead.

    Every painter must also implement :meth:`to_object` so it can be serialized
    and reconstructed by :meth:`deserialize`.
    """

    warning: str | None

    @staticmethod
    def deserialize(class_name: str, data: dict[str, Any]) -> "Painter":
        # Imported lazily to avoid a cycle between the base class and the
        # registry of concrete painter classes.
        from . import ALL_PAINTERS, PAINTERS_INDEX

        assert class_name in PAINTERS_INDEX
        painter_class = ALL_PAINTERS[PAINTERS_INDEX[class_name]]
        return painter_class(**data)

    @abstractmethod
    def paint(
        self,
        points: NDArray[np.complex128],
        ans: NDArray[np.uint32],
    ) -> None:
        """Assigns a palette index to each ordinary-precision complex point.

        ``points`` is a one-dimensional array of shape ``(n,)`` whose entries
        are complex plane coordinates, normally represented as
        ``numpy.complex128`` values. ``ans`` is a writable one-dimensional
        ``numpy.uint32`` array with the same shape. For every ``i``, the
        painter must write the non-negative palette index for ``points[i]`` to
        ``ans[i]``. The renderer later maps these indices to colors; indices
        beyond the palette length wrap around modulo that length.

        Implementations should fill all entries of ``ans`` in place and return
        ``None``. They may set ``self.warning`` to a user-facing message when
        rendering encounters a recoverable problem.
        """

    @abstractmethod
    def to_object(self) -> dict[str, Any]:
        """Return constructor keyword arguments that recreate this painter."""


class HighPrecisionPainter(Painter):
    """Base class for painters that consume high-precision digit arrays.

    Subclasses implement :meth:`paint_high_precision` rather than :meth:`paint`.
    """

    def paint(
        self,
        points: NDArray[np.complex128],
        ans: NDArray[np.uint32],
    ) -> None:
        raise NotImplementedError("Call paint_high_precision for this painter")

    @abstractmethod
    def paint_high_precision(
        self,
        points_x: NDArray[np.int64],
        points_y: NDArray[np.int64],
        ans: NDArray[np.uint32],
    ) -> None:
        """Assigns a palette index to each high-precision complex point.

        ``points_x`` and ``points_y`` are two-dimensional ``numpy.int64``
        arrays of identical shape ``(n, precision)``. Row ``i`` of each array
        contains the base-10**8 digits of an :class:`Hpn`: column zero is the
        integer part and subsequent columns are fractional digits. Together,
        the two rows encode the complex coordinate
        ``Hpn(points_x[i]) + 1j * Hpn(points_y[i])``.

        ``ans`` is a writable one-dimensional ``numpy.uint32`` array of shape
        ``(n,)``. For every point ``i``, the painter must write its non-negative
        palette index to ``ans[i]``. The renderer subsequently maps indices to
        colors, wrapping indices modulo the palette length.

        Implementations should fill all entries of ``ans`` in place and return
        ``None``. They may normalize or otherwise use the coordinate arrays as
        working storage, because the renderer does not reuse them after this
        call, and may set ``self.warning`` for recoverable rendering problems.
        """
