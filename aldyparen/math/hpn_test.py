from aldyparen.math.hpn import *
import pytest

CANONICAL_NUMBERS = ["0.", "1.", "-1.", "0.1", "0.001", "-0.001", "3.14", "0.99", "-2.15"]
NON_CANONICAL_NUMBERS = [("1e5", "100000."),
                         ("1e-5", "0.00001"),
                         ("-1.2e-10", "-0.00000000012"),
                         ("1.2e10", "12000000000.")]
LONG_NUMBERS = ["3.141592653589793238462643383279502884",
                "-283682.2979283798789737738828737364555555555555555555555518167282784268746267467466666",
                "0.0000000000082678678634782647862374628364823648263846238468268263863482648726386428674",
                "0." + "0" * 20 + "1" + "0" * 20 + "2",
                "-0." + "0" * 20 + "1" + "0" * 20 + "2"]
ALL_TEST_NUMBERS = CANONICAL_NUMBERS + [x[0] for x in NON_CANONICAL_NUMBERS] + LONG_NUMBERS


def test_to_from_string():
    for x in CANONICAL_NUMBERS:
        assert str(Hpn.from_str(x)) == x
    for number, canonical in NON_CANONICAL_NUMBERS:
        assert str(Hpn.from_str(number)) == canonical


def test_arithmetic():
    numbers_hpn = [Hpn.from_str(x, prec=16) for x in ALL_TEST_NUMBERS]
    numbers_np = [np.double(x) for x in ALL_TEST_NUMBERS]
    n = len(ALL_TEST_NUMBERS)

    for i in range(n):
        x = numbers_hpn[i]
        for j in range(n):
            y = numbers_hpn[j]
            assert np.allclose((x + y).to_float(), numbers_np[i] + numbers_np[j])
            assert np.allclose((x - y).to_float(), numbers_np[i] - numbers_np[j])
            if abs(x.digits[0]) < 1e5 and abs(y.digits[0]) < 1e5:
                assert np.allclose((x * y).to_float(), numbers_np[i] * numbers_np[j])


def test_conversion_precision_errors():
    with pytest.raises(Exception) as e:
        Hpn.from_str("0.0001", prec=1000)
    assert "Precision too large" in str(e)

    with pytest.raises(Exception) as e:
        Hpn.from_str("0.11111111111111111111111111111111111111", prec=2)
    assert "Insufficient precision" in str(e)

    with pytest.raises(Exception) as e:
        Hpn.from_str("0.0001e-1000", prec=16)
    assert "Insufficient precision" in str(e)


def test_conversion_range():
    min_value = -2 ** 63
    max_value = 2 ** 63 - 1
    assert np.allclose(Hpn.from_str(str(min_value)).to_float(), min_value)
    assert np.allclose(Hpn.from_str(str(max_value)).to_float(), max_value)

    # Note: error message for "int too large" is different on different platforms.
    with pytest.raises(Exception):
        Hpn.from_str(str(max_value + 1))
    with pytest.raises(Exception):
        Hpn.from_str(str(min_value - 1))

    with pytest.raises(Exception) as e:
        Hpn.from_str("12e100000000")
    assert "Exponent too large" in str(e)

    with pytest.raises(Exception) as e:
        Hpn.from_str("-10000000000e18")
    assert "Integer part outside of int64 range" in str(e)

    assert np.allclose(Hpn.from_str("0.00001e23").to_float(), 1e18)

    with pytest.raises(Exception) as e:
        Hpn.from_str("0.00001e24")
    assert "Integer part outside of int64 range" in str(e)


def test_conversion_vec():
    a = np.random.random(10)
    a_hpn = hpn_from_numpy_vec(a)
    a2 = hpn_to_numpy_vec(a_hpn)
    assert np.allclose(a, a2)


def test_vector_mul():
    # Test vector (point-wise) multiplication.
    n = 20
    a = np.random.random(n)
    b = np.random.random(n)

    a_hpn = hpn_from_numpy_vec(a)
    b_hpn = hpn_from_numpy_vec(b)
    c_hpn = np.empty_like(a_hpn)
    c = a * b
    hpn_mul_vec_inplace(a_hpn, b_hpn, c_hpn)
    assert np.allclose(c, hpn_to_numpy_vec(c_hpn))
