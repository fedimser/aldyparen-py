"""Library of functions operating on high-precision complex numbers (HPCNs).

HPCNs are represented as Numba tuples of 2 HPNs.
"""
import numba
import numpy as np

from aldyparen.math.hpn import hpn_normalize_in_place, Hpn, hpn_mul_inplace_noclear, hpn_abs, \
    HPN_TYPE, HPN_MUT, DEFAULT_PRECISION

# Numba types
HPCN_TYPE = numba.types.UniTuple(HPN_TYPE, 2)
HPCN_MUT = numba.types.UniTuple(HPN_MUT, 2)


class ComplexHpn:
    def __init__(self, real: Hpn, imag: Hpn):
        self.real = real
        self.imag = imag

    @staticmethod
    def from_number(x: int | float | complex, prec=DEFAULT_PRECISION) -> "ComplexHpn":
        x = complex(x)
        return ComplexHpn(Hpn.from_number(x.real, prec=prec), Hpn.from_number(x.imag, prec=prec))

    def to_complex(self) -> np.complex128:
        return np.complex128(self.real.to_float() + 1j * self.imag.to_float())

    @staticmethod
    def from_raw(raw: tuple[np.ndarray, np.ndarray]) -> "ComplexHpn":
        return ComplexHpn(Hpn(raw[0]), Hpn(raw[1]))

    def to_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns tuple that can be used directly in computations."""
        return self.real.digits, self.imag.digits

    def __add__(self, other: "ComplexHpn"):
        return ComplexHpn.from_raw(add(self.to_raw(), other.to_raw()))

    def __sub__(self, other: "ComplexHpn"):
        return ComplexHpn.from_raw(sub(self.to_raw(), other.to_raw()))

    def __mul__(self, other: "ComplexHpn"):
        return ComplexHpn.from_raw(mul(self.to_raw(), other.to_raw()))


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
