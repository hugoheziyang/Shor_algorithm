from fractions import Fraction

from shor_common.classical_post import (
    bits_to_fraction,
    continued_fraction_expansion,
    convergents_from_cf,
    convergents,
    recover_order_from_fraction_one_shot,
    recover_order_from_bits_one_shot,
    reduce_to_order,
    recover_order_from_fraction,
    recover_order_from_bits,
    shor_factors_from_order,
)


def test_bits_to_fraction():
    assert bits_to_fraction([1, 0, 1]) == Fraction(5, 8)
    assert bits_to_fraction([0, 1, 0, 0]) == Fraction(1, 4)


def test_continued_fraction_expansion():
    assert continued_fraction_expansion(Fraction(11, 16)) == [0, 1, 2, 5]
    assert continued_fraction_expansion(Fraction(1, 4)) == [0, 4]


def test_convergents_from_cf():
    cf = [0, 4]
    convs = convergents_from_cf(cf)
    assert convs == [Fraction(0, 1), Fraction(1, 4)]


def test_convergents_wrapper():
    convs = convergents(Fraction(11, 16))
    assert convs[-1] == Fraction(11, 16)


def test_reduce_to_order():
    # For a=2 mod 15, the true order is 4.
    # 8 is also a valid exponent, but should reduce to 4.
    assert reduce_to_order(2, 8, 15) == 4


def test_recover_order_from_fraction():
    r = recover_order_from_fraction(Fraction(1, 4), a=2, N=15)
    assert r == 4


def test_recover_order_from_fraction_reduced_phase():
    # 1/2 corresponds to 2/4, so the denominator alone is not the true order.
    # The routine should recover the true order by checking multiples.
    r = recover_order_from_fraction(Fraction(1, 2), a=2, N=15)
    assert r == 4


def test_recover_order_from_fraction_one_shot():
    r = recover_order_from_fraction_one_shot(Fraction(1, 4), a=2, N=15)
    assert r == 4


def test_recover_order_from_fraction_one_shot_reduced_phase_returns_none():
    # A single hardware-like shot only reveals denominator 2 here, not the
    # full order 4, so post-processing should fail cleanly and rely on a rerun.
    r = recover_order_from_fraction_one_shot(Fraction(1, 2), a=2, N=15)
    assert r is None


def test_recover_order_from_fraction_one_shot_checks_all_bounded_convergents():
    # The convergents of 2/9 include 1/4 and then 2/9. For a=2 mod 15 the true
    # order is 4, so a one-shot routine should not insist on the last bounded
    # convergent if an earlier bounded convergent already verifies.
    r = recover_order_from_fraction_one_shot(Fraction(2, 9), a=2, N=15)
    assert r == 4


def test_recover_order_from_bits():
    bits = [0, 1, 0, 0]  # 0.0100_2 = 1/4
    r = recover_order_from_bits(bits, a=2, N=15)
    assert r == 4


def test_recover_order_from_bits_one_shot():
    bits = [0, 1, 0, 0]  # 0.0100_2 = 1/4
    r = recover_order_from_bits_one_shot(bits, a=2, N=15)
    assert r == 4


def test_recover_order_from_bits_one_shot_reduced_phase_returns_none():
    bits = [1, 0, 0, 0]  # 0.1000_2 = 1/2
    r = recover_order_from_bits_one_shot(bits, a=2, N=15)
    assert r is None


def test_shor_factors_from_order():
    factors = shor_factors_from_order(a=2, r=4, N=15)
    assert factors in [(3, 5), (5, 3)]


def test_shor_factors_from_order_rejects_odd_order():
    assert shor_factors_from_order(a=2, r=3, N=15) is None
