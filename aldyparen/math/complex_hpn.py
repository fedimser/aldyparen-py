"""Library of functions operating on high-precision complex numbers (HPCNs).

HPCNs are represented as Numba tuples of 2 HPNs.
"""
import functools
from dataclasses import dataclass

import numba
import numpy as np

from aldyparen.math.hpn import hpn_normalize_in_place, Hpn, hpn_mul_inplace_noclear, hpn_abs, \
    HPN_TYPE, HPN_MUT, DEFAULT_PRECISION

# Numba types
HPCN_TYPE = numba.types.UniTuple(HPN_TYPE, 2)
HPCN_MUT = numba.types.UniTuple(HPN_MUT, 2)


@dataclass(frozen=True)
class ComplexHpn:
    real: Hpn
    imag: Hpn

    def __post_init__(self):
        assert self.real.prec() == self.imag.prec()

    @staticmethod
    def from_number(x: int | float | complex | np.complex128, prec=None) -> "ComplexHpn":
        return ComplexHpn.from_complex(complex(x), prec=prec)

    @staticmethod
    def from_complex(x: complex | np.complex128, prec=None) -> "ComplexHpn":
        x, y = Hpn.equalize_precisions([Hpn.create(x.real), Hpn.create(x.imag)], min_prec=prec or 0)
        return ComplexHpn(x, y)

    @functools.cached_property
    def approx(self) -> np.complex128:
        return np.complex128(self.real.to_float() + 1j * self.imag.to_float())

    @staticmethod
    def from_raw(raw: tuple[np.ndarray, np.ndarray]) -> "ComplexHpn":
        return ComplexHpn(Hpn(raw[0]), Hpn(raw[1]))

    def to_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns tuple that can be used directly in computations."""
        return self.real.digits, self.imag.digits

    def prec(self) -> int:
        return self.real.prec()

    def extend_precision(self, new_prec: int) -> 'ComplexHpn':
        return ComplexHpn(self.real.extend_precision(new_prec), self.imag.extend_precision(new_prec))

    def _prepare_binary_op(self, other: 'ComplexHpn') -> tuple['ComplexHpn', 'ComplexHpn']:
        if type(other) is not ComplexHpn:
            other = ComplexHpn.from_number(other)
        if other.prec() < self.prec():
            return self, other.extend_precision(self.prec())
        elif other.prec() > self.prec():
            return self.extend_precision(other.prec()), other
        return self, other

    def __add__(self, other: "ComplexHpn"):
        x, y = self._prepare_binary_op(other)
        return ComplexHpn.from_raw(add(x.to_raw(), y.to_raw()))

    def __sub__(self, other: "ComplexHpn"):
        x, y = self._prepare_binary_op(other)
        return ComplexHpn.from_raw(sub(x.to_raw(), y.to_raw()))

    def __mul__(self, other: "ComplexHpn"):
        x, y = self._prepare_binary_op(other)
        return ComplexHpn.from_raw(mul(x.to_raw(), y.to_raw()))


@numba.jit(HPCN_MUT(HPCN_TYPE, HPCN_TYPE), nopython=True)
def add(x, y):
    """Multiply two HPCNs."""
    ans_real = x[0] + y[0]
    ans_imag = x[1] + y[1]
    hpn_normalize_in_place(ans_real)
    hpn_normalize_in_place(ans_imag)
    return ans_real, ans_imag


@numba.jit(HPCN_MUT(HPCN_TYPE, HPCN_TYPE), nopython=True)
def sub(x, y):
    """Multiply two HPCNs."""
    ans_real = x[0] - y[0]
    ans_imag = x[1] - y[1]
    hpn_normalize_in_place(ans_real)
    hpn_normalize_in_place(ans_imag)
    return ans_real, ans_imag


# Warning: this can overflow for prec>450. To avoid, need to normalize after each multiplication.
@numba.jit(HPCN_MUT(HPCN_TYPE, HPCN_TYPE), nopython=True)
def mul(x, y):
    """Multiply two HPCNs."""
    ans_real = np.zeros_like(x[0])
    ans_imag = np.zeros_like(x[0])
    hpn_mul_inplace_noclear(x[1], y[1], ans_real)
    ans_real *= -1
    hpn_mul_inplace_noclear(x[0], y[0], ans_real)
    hpn_mul_inplace_noclear(x[0], y[1], ans_imag)
    hpn_mul_inplace_noclear(x[1], y[0], ans_imag)
    hpn_normalize_in_place(ans_real)
    hpn_normalize_in_place(ans_imag)
    return ans_real, ans_imag


@numba.jit(HPCN_MUT(HPCN_TYPE), nopython=True)
def sqr(x):
    """Square HPCN."""
    ans_real = np.zeros_like(x[0])
    ans_imag = np.zeros_like(x[0])
    hpn_mul_inplace_noclear(x[1], x[1], ans_real)
    ans_real *= -1
    hpn_mul_inplace_noclear(x[0], x[0], ans_real)
    hpn_mul_inplace_noclear(x[0], x[1], ans_imag)
    ans_imag *= 2
    hpn_normalize_in_place(ans_real)
    hpn_normalize_in_place(ans_imag)
    return ans_real, ans_imag


@numba.jit(HPCN_MUT(HPCN_TYPE, numba.types.int64), nopython=True)
def power_int(x, y):
    """Raises HPCN to integer power (using binary exponentiation)."""
    a = np.zeros_like(x[0]), np.zeros_like(x[1])
    a[0][0] = 1
    while True:
        if y & 1:
            a = mul(a, x)
        y = y >> 1
        if y == 0:
            break
        x = sqr(x)
    return a


@numba.jit(HPCN_MUT(HPCN_TYPE), nopython=True)
def abscw(x):
    """Component-wise modulus."""
    return hpn_abs(x[0]), hpn_abs(x[1])


@numba.jit(numba.types.boolean(HPCN_TYPE, HPN_TYPE), nopython=True)
def is_on_or_outside_circle(z, radius_squared):
    """Returns abs(z)>radius."""
    buf = np.zeros_like(radius_squared)
    hpn_mul_inplace_noclear(z[0], z[0], buf)
    hpn_mul_inplace_noclear(z[1], z[1], buf)
    buf -= radius_squared
    hpn_normalize_in_place(buf)
    return buf[0] >= 0
