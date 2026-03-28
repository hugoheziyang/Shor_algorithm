from fractions import Fraction
from math import gcd


def bits_to_fraction(bits: list[int]) -> Fraction:
    '''
    Convert a list of bits [b1, b2, ..., b_t] to the corresponding binary
    fraction sum_j b_j / 2^j.
    '''
    numerator = 0
    for bit in bits:
        numerator = numerator * 2 + int(bit) 

    denominator = 2 ** len(bits) 
    return Fraction(numerator, denominator)


def continued_fraction_expansion(x: Fraction) -> list[int]:
    '''
    Return the continued fraction expansion of a rational number x.

    The expansion is returned as a list of integers [a0, a1, a2, ...] such that
    x = a0 + 1/(a1 + 1/(a2 + ...)))
    '''
    if x < 0:
        raise ValueError("x must be nonnegative")

    cf = []
    n = x.numerator
    d = x.denominator

    while d != 0:
        a = n // d
        cf.append(a)
        n, d = d, n - a * d

    return cf


def convergents_from_cf(cf: list[int]) -> list[Fraction]:
    '''
    Given a continued fraction expansion, return the list of convergents.

    The k-th convergent is the rational number obtained by truncating the 
    continued fraction expansion after k terms.
    '''
    convs = []

    p_minus_2, p_minus_1 = 0, 1
    q_minus_2, q_minus_1 = 1, 0

    for a in cf:
        p = a * p_minus_1 + p_minus_2
        q = a * q_minus_1 + q_minus_2
        convs.append(Fraction(p, q))

        p_minus_2, p_minus_1 = p_minus_1, p
        q_minus_2, q_minus_1 = q_minus_1, q

    return convs


def convergents(x: Fraction) -> list[Fraction]:
    '''
    Return the list of convergents for a rational number x.
    '''
    return convergents_from_cf(continued_fraction_expansion(x))


def reduce_to_order(a: int, candidate_r: int, N: int) -> int:
    '''
    Reduce a candidate exponent to the actual order by removing prime factors.
    '''
    if candidate_r <= 0:
        raise ValueError("candidate_r must be positive")

    if pow(a, candidate_r, N) != 1:
        raise ValueError("candidate_r is not a valid exponent")

    r = candidate_r
    factor = 2

    # Only need to check prime factors up to sqrt(r) since if 
    # r = x * y and x > sqrt(r) then y < sqrt(r)
    while factor * factor <= r:
        while r % factor == 0 and pow(a, r // factor, N) == 1:
            r //= factor
        factor += 1

    return r


def recover_order_from_fraction(phase_frac: Fraction, a: int, N: int, 
                                max_multiple: int | None = None) -> int | None:
    '''
    Recover the order r from the estimated phase fraction.
    '''
    if not (0 <= phase_frac < 1):
        raise ValueError("phase_frac must satisfy 0 <= phase_frac < 1")
    
    # Constrain order r to be at most max_multiple * q, where q is the denominator of the convergent.
    if max_multiple is None:
        max_multiple = N

    for frac in convergents(phase_frac):
        q = frac.denominator

        if q <= 0:
            continue

        if pow(a, q, N) == 1:
            return reduce_to_order(a, q, N)

        for k in range(2, max_multiple + 1):
            candidate_r = k * q
            if pow(a, candidate_r, N) == 1:
                return reduce_to_order(a, candidate_r, N)

    return None


def recover_order_from_bits(bits: list[int], a: int, N: int, 
                            max_multiple: int | None = None) -> int | None:
    '''
    Recover the order r from the measured bits of the phase.
    '''
    phase_frac = bits_to_fraction(bits)
    return recover_order_from_fraction(phase_frac, a, N, max_multiple=max_multiple)


def shor_factors_from_order(a: int, r: int | None, N: int) -> tuple[int, int] | None:
    '''
    Given a candidate order r, attempt to recover the factors of N.

    If r is even and a^(r/2) is not congruent to ±1 mod N, then we can compute
    the factors as gcd(a^(r/2) ± 1, N). If these factors are nontrivial, 
    return them as a tuple (p, q). Otherwise, return None.
    '''
    if r is None or r % 2 != 0:
        return None

    x = pow(a, r // 2, N)

    if x == 1 or x == N - 1:
        return None

    p = gcd(x - 1, N)
    q = gcd(x + 1, N)

    if 1 < p < N:
        return (p, N // p)

    if 1 < q < N:
        return (q, N // q)

    return None