"""High-precision numbers (HPNs).

Representation
 * A HPN is represented by a NumPy array `a` of type (signed) int64 and length `prec`.
 * a[0] represents the integer part.
 * a[1:] represent digits of fractional part in base-(10^8).
   So, they take value (when normalized) in range [0, 10^8-1].
 * Vector `a` represents number `sum_i a[i]*(10^(-8*i))`.
 * HPN can represent a number in range (2^63-1, 2^63). This is checked when number is constructed. However,
   if an overflow happens during a calculation, it will be silently ignored and the result will be incorrect.
 * Maximum precision is 900, which is 7200 decimal digits. With this precision it's guaranteed that we won't have an
   overflow in multiplication (because 1e8*1e8*900 = 9e18 < 2^63).

Calculations:
 * Supported arithmetic: addition, subtraction, multiplication.
 * Can either use operators with `Hpn` objects (will extend to match precision or normalize), or operate directly
   on arrays (precision must match, must call `normalize` manually). The latter is intended to be called from Numba
   code.
 * Supports operations with 1D vectors of HPNs (all must have the same precision).
"""

import re
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

DIG_GROUP_LENGTH = 8
DIG_RANGE = 10 ** DIG_GROUP_LENGTH
MAX_PRECISION = 900
NUMBER_PATTERN = re.compile(r"^([-]?\d+)([.](\d*))?(e([-+]?\d+))?$")
DEFAULT_PRECISION = 8

# Numba types
HPN_TYPE = numba.types.Array(numba.types.int64, 1, 'A', readonly=True)
HPN_MUT = numba.types.Array(numba.types.int64, 1, 'A')


@dataclass(frozen=True)
class Hpn:
    digits: np.ndarray  # Point displayed at the center of the frame.

    @staticmethod
    def create(value: Any) -> "Hpn":
        if type(value) is Hpn:
            return value
        if type(value) is np.ndarray:
            assert type(value) is np.ndarray
            assert value.dtype == np.int64
            assert len(value.shape) == 1
            return Hpn(value)
        if type(value) is str:
            return Hpn(_hpn_from_str(value))
        return Hpn(_hpn_from_str(str(value)))

    def prec(self) -> int:
        return len(self.digits)

    def to_float(self) -> float:
        return _hpn_to_float(self.digits)

    def extend_precision(self, new_prec) -> 'Hpn':
        old_prec = len(self.digits)
        assert new_prec >= old_prec
        if new_prec > old_prec:
            return Hpn(np.pad(self.digits, (0, new_prec - old_prec), 'constant', constant_values=(0, 0)))
        return self

    def _prepare_binary_op(self, other: 'Hpn') -> tuple['Hpn', 'Hpn']:
        if other.prec() < self.prec():
            return self, other.extend_precision(self.prec())
        elif other.prec() > self.prec():
            return self.extend_precision(other.prec()), other
        return self, other

    def __add__(self, other: 'Hpn'):
        x, y = self._prepare_binary_op(other)
        ans = x.digits + y.digits
        hpn_normalize_in_place(ans)
        return Hpn(ans)

    def __sub__(self, other: 'Hpn'):
        x, y = self._prepare_binary_op(other)
        ans = x.digits - y.digits
        hpn_normalize_in_place(ans)
        return Hpn(ans)

    def __mul__(self, other: 'Hpn'):
        x, y = self._prepare_binary_op(other)
        ans = hpn_mul(x.digits, y.digits)
        hpn_normalize_in_place(ans)
        return Hpn(ans)

    def __str__(self):
        return _hpn_to_str(self.digits)

    @staticmethod
    def from_str(s: str, prec: int = None, extra_power_10=0) -> 'Hpn':
        """Creates HPN from string representation with given precision.
        :param prec: Precision.
        :param extra_power_10: Multiplies result by 10^extra_power_10.
        """
        return Hpn(_hpn_from_str(s, prec=prec, extra_power_10=extra_power_10))

    @staticmethod
    def from_number(value: int | float, prec: int = DEFAULT_PRECISION) -> 'Hpn':
        """Creates HPN from number."""
        return Hpn.from_str(str(value), prec=prec)

    @staticmethod
    def equalize_precisions(args: list['Hpn'], min_prec=2) -> list['Hpn']:
        prec = min_prec
        for arg in args:
            prec = max(prec, arg.prec())
        return [Hpn.extend_precision(arg, prec) for arg in args]


@numba.jit("void(i8[:])", nopython=True, inline="always")
def hpn_normalize_in_place(x):
    prec = x.shape[0]
    for i in range(prec - 1, 0, -1):
        x[i - 1] += x[i] // DIG_RANGE
        x[i] %= DIG_RANGE


@numba.jit("void(i8[:,:])", nopython=True)
def hpn_normalize_in_place_vec(x):
    prec = x.shape[1]
    for i in range(prec - 1, 0, -1):
        x[:, i - 1] += x[:, i] // DIG_RANGE
        x[:, i] %= DIG_RANGE


def _hpn_from_str(s, prec: int = None, extra_power_10=0) -> np.ndarray:
    match = NUMBER_PATTERN.match(s)
    assert match is not None, f"Invalid syntax: {s}"
    int_part, _, frac_part, _, exp_part = match.groups()
    exp_val = int(exp_part) if exp_part is not None else 0
    exp_val += extra_power_10

    if frac_part is None:
        frac_part = ""
    if len(frac_part) % DIG_GROUP_LENGTH != 0:
        frac_part += "0" * (DIG_GROUP_LENGTH - (len(frac_part) % DIG_GROUP_LENGTH))
    assert len(frac_part) % DIG_GROUP_LENGTH == 0
    frac_digits = len(frac_part) // DIG_GROUP_LENGTH
    need_precision = max(2, 1 + frac_digits + int(np.ceil(max(0, -exp_val) / DIG_GROUP_LENGTH)))
    if prec is None:
        prec = need_precision
    assert prec >= need_precision, f"Insufficient precision, need at least {need_precision}"
    assert prec <= MAX_PRECISION, "Precision too large"

    result = np.zeros(prec, dtype=np.int64)
    result[0] = np.int64(int_part)
    for i in range(frac_digits):
        result[i + 1] = int(frac_part[i * DIG_GROUP_LENGTH: (i + 1) * DIG_GROUP_LENGTH])
    if int_part[0] == '-':
        result[1:] *= -1

    if exp_val < 0:
        shift_right = -(exp_val // DIG_GROUP_LENGTH)
        result[shift_right:] = result[0:-shift_right]
        result[:shift_right] = 0
        exp_val %= DIG_GROUP_LENGTH

    assert exp_val >= 0
    if exp_val > 0:
        hpn_normalize_in_place(result)
        while exp_val >= DIG_GROUP_LENGTH and result[0] == 0:
            exp_val -= DIG_GROUP_LENGTH
            result = np.roll(result, -1)
        if exp_val > 20:
            raise ValueError("Exponent too large")
        mul_factor = 10 ** exp_val
        if abs(int(result[0]) * mul_factor) >= (2 ** 63):
            raise ValueError("Integer part outside of int64 range")
        result *= mul_factor

    hpn_normalize_in_place(result)
    return result


def _hpn_from_number(x, prec=DEFAULT_PRECISION) -> np.ndarray:
    """Creates HPN from number (can be any numeric type)."""
    return _hpn_from_str(str(x), prec=prec)


def hpn_from_numpy_vec(x, prec=DEFAULT_PRECISION):
    """Creates vector of HPNs from vector of numbers."""
    assert len(x.shape) == 1
    return np.array([_hpn_from_str(str(num), prec=prec) for num in x], dtype=np.int64)


def hpn_to_numpy_vec(x: np.ndarray) -> np.ndarray:
    """Converts HPN vector to vector of doubles."""
    n, prec = x.shape
    ans = np.zeros((n,), dtype=np.double)
    k = np.double(1)
    for i in range(prec):
        ans[:] += x[:, i] * k
        k /= DIG_RANGE
    return ans


def _frac_to_str(x):
    return "".join(str(x).zfill(DIG_GROUP_LENGTH) for x in x[1:]).rstrip("0")


def _hpn_to_str(x) -> str:
    """String representation of HPN, with full precision."""
    hpn_normalize_in_place(x)
    if x[0] < 0 and not np.all(x[1:] == 0):
        t = np.zeros_like(x) - x.copy()
        hpn_normalize_in_place(t)
        ans = ("-0" if x[0] == -1 else str(x[0] + 1)) + "." + _frac_to_str(t)
    else:
        ans = str(x[0]) + "." + _frac_to_str(x)
    return ans


def _hpn_to_float(x: np.ndarray) -> float:
    """Converts HPN to float (with precision loss)."""
    return float(_hpn_to_str(x))


@numba.jit("i8[:](i8[:],i8[:])", nopython=True)
def hpn_mul(x, y):
    prec = x.shape[0]
    ans = np.zeros_like(x)
    for i in range(prec):
        ans[i:] += x[i] * y[:prec - i]
    return ans


@numba.jit(numba.types.void(HPN_TYPE, HPN_TYPE, HPN_MUT), nopython=True)
def hpn_mul_inplace_noclear(x, y, ans):
    prec = x.shape[0]
    for i in range(prec):
        ans[i:] += x[i] * y[:prec - i]


@numba.jit("void(i8[:,:],i8[:,:],i8[:,:])", parallel=True, nopython=True)
def hpn_mul_vec_inplace(x, y, ans):
    n, prec = x.shape
    ans[:] = 0
    for j in numba.prange(n):
        for i in range(prec):
            ans[j, i:] += x[j, i] * y[j, :prec - i]


@numba.jit("i8[:](i8[:])", nopython=True)
def hpn_square(x):
    ans = hpn_mul(x, x)
    hpn_normalize_in_place(ans)
    return ans


@numba.jit(HPN_MUT(HPN_TYPE), nopython=True)
def hpn_abs(x):
    if x[0] >= 0:
        return np.copy(x)
    ans = x * -1
    hpn_normalize_in_place(ans)
    return ans
