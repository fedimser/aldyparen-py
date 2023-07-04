from aldyparen.math.hpn import *

CANONICAL_NUMBERS = ["0.", "1.", "-1.", "0.1", "0.001", "-0.001", "3.14", "0.99", "-2.15"]
NON_CANONICAL_NUMBERS = [("1e5", "100000."),
                         ("1e-5", "0.00001"),
                         ("-1.2e-10", "-0.00000000012"),
                         ("1.2e10", "12000000000.")]
LONG_NUMBERS = ["3.141592653589793238462643383279502884",
                "0." + "0" * 10 + "1" + "0" * 10 + "2",
                "-0." + "0" * 10 + "1" + "0" * 10 + "2"]
ALL_TEST_NUMBERS = CANONICAL_NUMBERS + [x[0] for x in NON_CANONICAL_NUMBERS] + LONG_NUMBERS


def test_to_from_string():
    for x in CANONICAL_NUMBERS:
        assert hpn_to_str(hpn_from_str(x)) == x
    for number, canonical in NON_CANONICAL_NUMBERS:
        assert hpn_to_str(hpn_from_str(number)) == canonical


def test_arithmetic():
    numbers_hpn = [hpn_from_str(x) for x in ALL_TEST_NUMBERS]
    numbers_np = [np.double(x) for x in ALL_TEST_NUMBERS]
    n = len(ALL_TEST_NUMBERS)

    def to_np(x):
        return np.double(hpn_to_str(x))

    for i in range(n):
        x = numbers_hpn[i]
        for j in range(n):
            y = numbers_hpn[j]
            assert np.allclose(to_np(x + y), numbers_np[i] + numbers_np[j])
            assert np.allclose(to_np(x - y), numbers_np[i] - numbers_np[j])
            if abs(x[0]) < 1e5 and abs(y[0]) < 1e5:
                assert np.allclose(to_np(hpn_mul(x, y)), numbers_np[i] * numbers_np[j])


"""
def test_vector_mul():
    # Test vector multiplication.
    n = 10
    a = np.random.random(n)
    b = np.random.random(n)
    a_hpn = hpn_from_numpy_vec(a)
    b_hpn = hpn_from_numpy_vec(b)
    a2 = a * a
    c = a * b
    a2_hpn = hpn_square_vec(a_hpn)
    c_hpn = hpn_mul_vec(a_hpn, b_hpn)
    assert np.allclose(a2, hpn_to_numpy_vec(a2_hpn))
    assert np.allclose(c, hpn_to_numpy_vec(c_hpn))
"""
