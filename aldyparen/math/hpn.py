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
 * We represent numbers directly as array without wrapping them in an object, so that they casn be used in Numba.

Calculations:
 * Supported arithmetic: addition (use "+"), subtraction (use "-"), multiplication.
 * Arithmetic can be done only with HPNs having the same precision.
 * Normalization is not done automatically, you need call `hpn_normalize_in_place` function.
 * Supports operations with 1D vectors of HPNs (all must have the same precision).
"""

import re

import numba
import numpy as np

DIG_GROUP_LENGTH = 8
DIG_RANGE = 10 ** DIG_GROUP_LENGTH
MAX_PRECISION = 900
NUMBER_PATTERN = re.compile(r"^([-]?\d+)([.](\d*))?(e([-+]?\d+))?$")


@numba.jit("void(i8[:])")
def hpn_normalize_in_place(x):
    prec = x.shape[0]
    for i in range(prec - 1, 0, -1):
        x[i - 1] += x[i] // DIG_RANGE
        x[i] %= DIG_RANGE


@numba.jit("void(i8[:,:])")
def hpn_normalize_in_place_vec(x):
    prec = x.shape[1]
    for i in range(prec - 1, 0, -1):
        x[:, i - 1] += x[:, i] // DIG_RANGE
        x[:, i] %= DIG_RANGE


def hpn_from_str(s, prec=16, extra_power_10=0) -> np.ndarray:
    """Creates HPN from string representation with given precision.
    :param prec: Precision.
    :param extra_power_10: Multiplies result by 10^extra_power_10.
    """
    assert prec >= 2
    assert prec <= MAX_PRECISION, "Precision too large"
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
    need_precision = 1 + frac_digits + int(np.ceil(max(0, -exp_val) / DIG_GROUP_LENGTH))
    assert need_precision <= prec, f"Insufficient precision, need at least {need_precision}"

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


def hpn_from_number(x: float | int, prec=16):
    """Creates HPN from number (can be any numeric type)."""
    return hpn_from_str(str(x), prec=prec)


def hpn_from_numpy_vec(x, prec=16):
    """Creates vector of HPNs from vector of numbers."""
    assert len(x.shape) == 1
    return np.array([hpn_from_str(str(num), prec=prec) for num in x], dtype=np.int64)


def hpn_to_float(x: np.ndarray) -> float:
    """Converts HPN vector to float."""
    return float(hpn_to_str(x))


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


ZERO = hpn_from_str("0.0")


def hpn_to_str(x):
    """String representation of HPN, with full precision."""
    hpn_normalize_in_place(x)
    if x[0] < 0 and not np.all(x[1:] == 0):
        t = ZERO - x.copy()
        hpn_normalize_in_place(t)
        ans = ("-0" if x[0] == -1 else str(x[0] + 1)) + "." + _frac_to_str(t)
    else:
        ans = str(x[0]) + "." + _frac_to_str(x)
    return ans


@numba.jit("i8[:](i8[:],i8[:])")
def hpn_mul(x, y):
    prec = x.shape[0]
    ans = np.zeros_like(x)
    for i in range(prec):
        ans[i:] += x[i] * y[:prec - i]
    return ans


@numba.jit("void(i8[:,:],i8[:,:],i8[:,:])", parallel=True)
def hpn_mul_vec_inplace(x, y, ans):
    n, prec = x.shape
    ans[:] = 0
    for j in numba.prange(n):
        for i in range(prec):
            ans[j, i:] += x[j, i] * y[j, :prec - i]
