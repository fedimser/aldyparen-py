# High precision computing.

import re

import numba
import numpy as np

GROUP_COUNT = 10  # 171
DIG_GROUP_LENGTH = 8
DIG_RANGE = 10 ** DIG_GROUP_LENGTH
NUMBER_PATTERN = re.compile(r"^([-]?\d+)([.](\d*))?(e([-]?\d+))?$")


# Contract.
# We store a[0..DIG_COUNT-1] of 64-bit signed integer.
# All except first one are by contract non-negative, in range [0..10^DIG_SIZE-1]
# Formally, `a` represents number sum_i a[i]*(DIG_RANGE**-i)


def hpn_normalize_in_place(x):
    for i in range(GROUP_COUNT - 1, 0, -1):
        x[i - 1] += x[i] // DIG_RANGE
        x[i] %= DIG_RANGE


@numba.jit("i8[:](i8[:])")
def hpn_normalize(x):
    for i in range(GROUP_COUNT - 1, 0, -1):
        x[i - 1] += x[i] // DIG_RANGE
        x[i] %= DIG_RANGE
    return x


"""
@numba.jit("i8[:,:](i8[:,:])", parallel=True)
def hpn_normalize_vec(x):
    for i in range(GROUP_COUNT - 1, 0, -1):
        x[:, i - 1] += x[:, i] // DIG_RANGE
        x[:, i] %= DIG_RANGE
    return x
"""


def hpn_from_str(s):
    match = NUMBER_PATTERN.match(s)
    assert match is not None, f"Invalid syntax: {s}"
    int_part, _, frac_part, _, exp_part = match.groups()

    if frac_part is None:
        frac_part = ""
    if len(frac_part) % DIG_GROUP_LENGTH != 0:
        frac_part += "0" * (DIG_GROUP_LENGTH - (len(frac_part) % DIG_GROUP_LENGTH))
    assert len(frac_part) % DIG_GROUP_LENGTH == 0
    frac_digits = len(frac_part) // DIG_GROUP_LENGTH
    assert frac_digits <= GROUP_COUNT - 1, f"Precision is not sufficient, need at least {frac_digits + 1}"

    result = np.zeros(GROUP_COUNT, dtype=np.int64)
    result[0] = int(int_part)
    for i in range(frac_digits):
        result[i + 1] = int(frac_part[i * DIG_GROUP_LENGTH: (i + 1) * DIG_GROUP_LENGTH])
    if int_part[0] == '-':
        result[1:] *= -1

    if exp_part is not None:
        # TODO: check overflow.
        exp = int(exp_part)
        result *= (10 ** (exp % DIG_GROUP_LENGTH))
        shift_left = exp // DIG_GROUP_LENGTH
        if shift_left < 0:
            shift_right = -shift_left
            result[shift_right:] = result[0:-shift_right]
            result[:shift_right] = 0
        elif shift_left > 0:
            result *= 10 ** (DIG_GROUP_LENGTH * shift_left)

    hpn_normalize_in_place(result)
    return result


def hpn_from_numpy_vec(x):
    assert len(x.shape) == 1
    n = x.shape[0]
    ans = np.zeros((n, GROUP_COUNT), dtype=np.int64)
    for i in range(n):
        ans[i, :] = hpn_from_str(str(x[i]))
    return ans


def hpn_to_numpy_vec(x):
    n = x.shape[0]
    assert x.shape == (n, GROUP_COUNT)
    ans = np.zeros((n,), dtype=np.double)
    k = np.double(1)
    for i in range(GROUP_COUNT):
        ans[:] += x[:, i] * k
        k /= DIG_RANGE
    return ans


# Ignores integer part.
def frac_to_str(x):
    ans = "".join(str(x).zfill(DIG_GROUP_LENGTH) for x in x[1:]).rstrip("0")
    return ans


ZERO = hpn_from_str("0.0")


def hpn_to_str(x):
    hpn_normalize_in_place(x)
    if x[0] < 0 and not np.all(x[1:] == 0):
        t = ZERO - x.copy()
        hpn_normalize_in_place(t)
        ans = ("-0" if x[0] == -1 else str(x[0] + 1)) + "." + frac_to_str(t)
    else:
        ans = str(x[0]) + "." + frac_to_str(x)
    return ans


@numba.jit("i8[:](i8[:],i8[:])")
def hpn_mul(x, y):
    ans = np.zeros_like(x)
    for i in range(GROUP_COUNT):
        ans[i:] += x[i] * y[:GROUP_COUNT - i]
    return ans


"""
@numba.jit("i8[:,:](i8[:,:],i8[:,:])", parallel=True)
def hpn_mul_vec(x, y):
    ans = np.zeros_like(x)
    for i in range(GROUP_COUNT):
        for j in range(GROUP_COUNT - i):
            ans[:, i + j] += x[:, i] * y[:, j]
    return ans
"""


@numba.jit("i8[:](i8[:])")
def hpn_square(x):
    ans = hpn_mul(x, x)
    return hpn_normalize(ans)


"""
@numba.jit("i8[:,:](i8[:,:])")
def hpn_square_vec(x):
    ans = hpn_mul_vec(x, x)
    return hpn_normalize_vec(ans)
"""
