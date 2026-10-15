"""Scale-independent angular uncertainty, with analytical dimensionless oracles."""

import math
from decimal import Decimal, localcontext

import pytest

from uncertainties import ufloat, umath


SCALES = [1.0, 1e155, 1e160, 1e200, 1e300, 1e-160, 1e-200, 1e-300]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("y0, x0", [(3, 4), (-3, 4), (3, -4), (-3, -4), (0, 4), (3, 0)])
def test_atan2_scale_independent_uncertainty(scale, y0, x0):
    y = ufloat(y0 * scale, 0.03 * scale)
    x = ufloat(x0 * scale, 0.04 * scale)
    angle = umath.atan2(y, x)

    # Work in the dimensionless coordinates, whose squared radius is small.
    radius_squared = x0**2 + y0**2
    dy = x0 / radius_squared
    dx = -y0 / radius_squared
    assert angle.n == math.atan2(y.n, x.n)
    assert math.isclose(angle.derivatives[y], dy / scale, rel_tol=1e-14, abs_tol=0)
    assert math.isclose(angle.derivatives[x], dx / scale, rel_tol=1e-14, abs_tol=0)
    assert math.isclose(angle.s, math.hypot(dy * 0.03, dx * 0.04), rel_tol=1e-14)


@pytest.mark.parametrize("scale", SCALES)
def test_atan2_scaled_correlated_coordinates(scale):
    displacement = ufloat(0, 0.01)
    x = scale * (4 + 2 * displacement)
    y = scale * (3 - displacement)
    angle = umath.atan2(y, x)
    # d atan2(3-t, 4+2t)/dt at t=0 is (-4 - 6)/25 = -0.4.
    assert math.isclose(angle.derivatives[displacement], -0.4, rel_tol=1e-14)
    assert math.isclose(angle.s, 0.004, rel_tol=1e-14)


@pytest.mark.parametrize("coordinate", [1e154, 1.7e308])
def test_atan2_overflowing_intermediate_radius(coordinate):
    y = ufloat(coordinate, coordinate * 0.01)
    x = ufloat(coordinate, coordinate * 0.01)
    angle = umath.atan2(y, x)
    # The squared radius (1e154) or radius itself (1.7e308) overflows,
    # but both partials and the angular standard deviation are representable.
    assert math.isclose(angle.derivatives[y], 0.5 / coordinate, rel_tol=1e-14)
    assert math.isclose(angle.derivatives[x], -0.5 / coordinate, rel_tol=1e-14)
    assert math.isclose(angle.s, math.sqrt(2) * 0.005, rel_tol=1e-14)


def test_unumpy_arctan2_scaled_coordinates():
    numpy = pytest.importorskip("numpy")
    from uncertainties import unumpy

    scales = numpy.array(SCALES)
    x = unumpy.uarray(4 * scales, 0.04 * scales)
    y = unumpy.uarray(3 * scales, 0.03 * scales)
    angles = unumpy.arctan2(y, x)
    numpy.testing.assert_allclose(unumpy.nominal_values(angles), math.atan2(3, 4))
    numpy.testing.assert_allclose(
        unumpy.std_devs(angles), math.sqrt(2) * 0.12 / 25, rtol=1e-14
    )


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_atan2_subnormal_minor_coordinate(swap, sign):
    # Normalizing the minor coordinate first rounds it in the subnormal range,
    # although the resulting partial is normal and the direct formula is safe.
    minor = sign * float.fromhex("0x0.0000000000001p-1022")
    major = 1.23456789e-8
    y0, x0 = (minor, major) if swap else (major, minor)
    with localcontext() as context:
        context.prec = 110
        dx, dy = Decimal.from_float(x0), Decimal.from_float(y0)
        denominator = dx * dx + dy * dy
        expected = float(-dy / denominator if swap else dx / denominator)
    y, x = ufloat(y0, 1e-10), ufloat(x0, 1e-10)
    angle = umath.atan2(y, x)
    variable = x if swap else y
    assert math.isclose(angle.derivatives[variable], expected, rel_tol=1e-14, abs_tol=0)


@pytest.mark.parametrize(
    "y0, x0",
    [
        (5e-324, 1e-155),
        (5e-324, 1e-160),
        (5e-324, 1e-162),
        (5e-309, 5e-309),
        (1e-200, 1e-154),
        (1e-200, 1e154),
        (1.7e308, 1.7e308),
    ],
)
def test_atan2_partial_safe_intermediates(y0, x0):
    from uncertainties.umath_core import fixed_derivatives

    with localcontext() as context:
        context.prec = 110
        dx, dy = Decimal.from_float(x0), Decimal.from_float(y0)
        denominator = dx * dx + dy * dy
        expected = [float(dx / denominator), float(-dy / denominator)]
    for derivative, reference in zip(fixed_derivatives["atan2"], expected):
        assert math.isclose(
            derivative(y0, x0), reference, rel_tol=1e-14, abs_tol=5e-324
        )


def test_atan2_origin_remains_undefined():
    angle = umath.atan2(ufloat(0, 0.01), ufloat(0, 0.01))
    assert math.isnan(angle.s)
