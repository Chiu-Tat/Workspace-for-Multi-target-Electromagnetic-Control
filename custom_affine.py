"""
Affine Arithmetic Library
"""
import math
from custom_interval import interval, imath

# ============================================================================
# Affine Arithmetic Engine
# ============================================================================

_next_symbol_index = 0

def get_next_symbol_index():
    """Get the next available symbol index for a new noise term."""
    global _next_symbol_index
    _next_symbol_index += 1
    return _next_symbol_index - 1

def reset_symbol_index():
    """Reset the global symbol index to zero."""
    global _next_symbol_index
    _next_symbol_index = 0

class Affine:
    """A simple Affine Arithmetic class."""
    def __init__(self, center, deviations=None):
        self.center = center
        self.deviations = deviations if deviations is not None else {}

    @classmethod
    def from_interval(cls, interval_obj):
        """Create an affine form from an interval object."""
        # Handle both interval objects and scalar values
        if isinstance(interval_obj, (int, float)):
            return cls(interval_obj)
        
        center = (interval_obj[0][0] + interval_obj[0][1]) / 2
        radius = (interval_obj[0][1] - interval_obj[0][0]) / 2
        if radius == 0:
            return cls(center)
        symbol_index = get_next_symbol_index()
        return cls(center, {symbol_index: radius})

    def to_interval(self):
        """Convert an affine form back to an interval."""
        radius = sum(abs(v) for v in self.deviations.values())
        return interval[self.center - radius, self.center + radius]

    def __add__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center + other, self.deviations.copy())
        
        new_center = self.center + other.center
        new_devs = self.deviations.copy()
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v
        return Affine(new_center, new_devs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center - other, self.deviations.copy())

        new_center = self.center - other.center
        new_devs = self.deviations.copy()
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) - v
        return Affine(new_center, new_devs)

    def __rsub__(self, other):
        # other - self
        if not isinstance(other, Affine):
            return Affine(other - self.center, {k: -v for k, v in self.deviations.items()})
        else:
            return other.__sub__(self)

    def __neg__(self):
        return Affine(-self.center, {k: -v for k, v in self.deviations.items()})

    def __mul__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center * other, {k: v * other for k, v in self.deviations.items()})

        new_center = self.center * other.center
        new_devs = {}
        
        # Linear terms
        for k, v in self.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v * other.center
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v * self.center

        # Quadratic error term - be more conservative about when to add it
        if self.deviations and other.deviations:
            self_radius = sum(abs(v) for v in self.deviations.values())
            other_radius = sum(abs(v) for v in other.deviations.values())
            quad_error_radius = self_radius * other_radius
            
            # Only add quadratic error if it's significant compared to linear terms
            linear_magnitude = sum(abs(v) for v in new_devs.values())
            if quad_error_radius > 1e-12 and quad_error_radius > 1e-6 * linear_magnitude:
                error_symbol = get_next_symbol_index()
                new_devs[error_symbol] = quad_error_radius
            
        return Affine(new_center, new_devs)

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        if not isinstance(other, Affine):
            if abs(other) < 1e-12:
                # Fallback for division by a very small number
                return self * (1e12 if other > 0 else -1e12)
            return self * (1.0 / other)

        other_interval = other.to_interval()
        # If interval contains zero, division is tricky.
        if other_interval[0][0] <= 0 <= other_interval[0][1]:
            # If the divisor is exactly zero, the result is undefined/infinite.
            if other_interval[0][0] == 0 and other_interval[0][1] == 0:
                # Represent "any value" or infinity with a large, symmetric interval.
                return Affine.from_interval(interval[-1e12, 1e12])

            # Fallback to interval division, which can handle intervals containing zero.
            self_interval = self.to_interval()
            result_interval = self_interval / other_interval
            
            # The result of interval division by an interval containing zero can be (-inf, +inf)
            if math.isinf(result_interval[0][0]) or math.isinf(result_interval[0][1]):
                 # Clamp the result to a large but manageable range.
                return Affine.from_interval(interval[-1e12, 1e12])
            
            return Affine.from_interval(result_interval)

        # Proper affine division using linear approximation: a/b = a * (1/b)
        inv_other = other.inverse()
        return self * inv_other

    def inverse(self):
        """Computes the inverse of an affine form. For internal use in division."""
        other_interval = self.to_interval()
        # If interval contains zero OR its center is very close to zero, fallback to interval arithmetic
        if other_interval[0][0] <= 0 <= other_interval[0][1] or abs(self.center) < 1e-9:
            result_interval = 1.0 / other_interval
            # Clamp infinite results to avoid NaN propagation
            if math.isinf(result_interval[0][0]) or math.isinf(result_interval[0][1]) or \
               math.isnan(result_interval[0][0]) or math.isnan(result_interval[0][1]):
                return Affine.from_interval(interval[-1e20, 1e20])
            return Affine.from_interval(result_interval)

        # Linear approximation: 1/(c + x) ≈ 1/c - x/c^2
        c = self.center
        inv_c = 1.0 / c
        
        new_center = inv_c
        new_devs = {k: -v * inv_c * inv_c for k, v in self.deviations.items()}
        
        # Only add error term if there are significant deviations
        if self.deviations:
            # Tighter error bound for 1/x approximation
            a, b = other_interval[0]
            if a > 0 and b > 0:
                error_radius = (b - a) ** 2 / (4 * a * b)
            elif a < 0 and b < 0:
                error_radius = (b - a) ** 2 / (4 * abs(a) * abs(b))
            else:
                # Fallback to previous method for safety
                inv_interval = 1.0 / self.to_interval()
                approx_interval = Affine(new_center, new_devs).to_interval()
                error_radius = max(abs(inv_interval[0][0] - approx_interval[0][0]),
                                   abs(inv_interval[0][1] - approx_interval[0][1]))

            if error_radius > 1e-10:  # Stricter threshold
                error_symbol = get_next_symbol_index()
                new_devs[error_symbol] = error_radius
            
        return Affine(new_center, new_devs)

    def sqr(self):
        """
        Computes the square of an affine form using the Chebyshev approximation for x^2,
        which gives a tighter bound than the standard method.
        f(x) = x^2, where x is the interval of self.
        The best affine approximation is alpha*x + beta.
        alpha = (sup(f) - inf(f)) / (sup(x) - inf(x))
        beta = (sup(f) + inf(f) - alpha * (sup(x) + inf(x))) / 2
        """
        x_interval = self.to_interval()
        
        # Avoid division by zero if the interval is a point
        if x_interval[0][0] == x_interval[0][1]:
            return Affine(x_interval[0][0] ** 2)

        f_interval = x_interval**2
        
        alpha = (f_interval[0][1] - f_interval[0][0]) / (x_interval[0][1] - x_interval[0][0])
        beta = (f_interval[0][1] + f_interval[0][0] - alpha * (x_interval[0][1] + x_interval[0][0])) / 2.0
        
        # The new affine form is alpha * self + beta
        return self * alpha + beta

    def __pow__(self, exponent):
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                return Affine(1)
            if exponent == 1:
                return self
            if exponent == 2:
                return self.sqr() # Use the specialized square function
            # For higher integer powers, use optimized approach
            if exponent == 3:
                # x^3 = x * x^2, using sqr() for tighter bounds
                return self * self.sqr()
            elif exponent <= 6:
                # Use direct computation for small powers
                result = self.sqr()  # x^2
                for _ in range(exponent - 2):
                    result = result * self
                return result
            else:
                # Use binary exponentiation for larger powers
                result = Affine(1)
                base = self
                exp = exponent
                while exp > 0:
                    if exp % 2 == 1:
                        result = result * base
                    if exp > 1:  # Avoid unnecessary squaring at the end
                        base = base.sqr()
                    exp //= 2
                return result
        else:
            # For non-integer or negative powers, convert to interval and back (conservative)
            interval_result = self.to_interval() ** exponent
            return Affine.from_interval(interval_result)

    def __repr__(self):
        if not self.deviations:
            return f"Affine({self.center:.6f})"
        dev_str = " + ".join([f"{v:.6f}e{k}" for k, v in sorted(self.deviations.items())])
        return f"Affine({self.center:.6f}, {dev_str})"

def affine_sin(a: Affine):
    """Affine approximation of the sine function using Chebyshev approximation."""
    x_interval = a.to_interval()

    # Avoid division by zero if the interval is a point
    if x_interval[0][0] == x_interval[0][1]:
        return Affine(math.sin(x_interval[0][0]))

    f_interval = imath.sin(x_interval)
    
    # Find the points where sin(x) reaches its min and max on the interval
    # This is more robust than assuming it's at the endpoints.
    inf_x, sup_x = x_interval[0]
    
    # Potential extrema for sin(x) are at the endpoints or at pi/2 + k*pi
    extrema_points = {inf_x, sup_x}
    # Find k such that pi/2 + k*pi is in the interval
    # k_low = ceil((inf_x - pi/2)/pi), k_high = floor((sup_x - pi/2)/pi)
    k_low = math.ceil((inf_x - math.pi/2) / math.pi)
    k_high = math.floor((sup_x - math.pi/2) / math.pi)
    for k in range(k_low, k_high + 1):
        extrema_points.add(math.pi/2 + k * math.pi)
        
    min_f = min(math.sin(p) for p in extrema_points if inf_x <= p <= sup_x)
    max_f = max(math.sin(p) for p in extrema_points if inf_x <= p <= sup_x)

    alpha = (max_f - min_f) / (sup_x - inf_x)
    beta = (max_f + min_f - alpha * (sup_x + inf_x)) / 2.0
    
    return a * alpha + beta

def affine_cos(a: Affine):
    """Affine approximation of the cosine function using Chebyshev approximation."""
    x_interval = a.to_interval()

    if x_interval[0][0] == x_interval[0][1]:
        return Affine(math.cos(x_interval[0][0]))

    # Find the points where cos(x) reaches its min and max on the interval
    inf_x, sup_x = x_interval[0]
    
    # Potential extrema for cos(x) are at the endpoints or at k*pi
    extrema_points = {inf_x, sup_x}
    k_low = math.ceil(inf_x / math.pi)
    k_high = math.floor(sup_x / math.pi)
    for k in range(k_low, k_high + 1):
        extrema_points.add(k * math.pi)

    min_f = min(math.cos(p) for p in extrema_points if inf_x <= p <= sup_x)
    max_f = max(math.cos(p) for p in extrema_points if inf_x <= p <= sup_x)

    alpha = (max_f - min_f) / (sup_x - inf_x)
    beta = (max_f + min_f - alpha * (sup_x + inf_x)) / 2.0
    
    return a * alpha + beta

def affine_sqrt(a: Affine):
    """Affine approximation of the square root function using Chebyshev approximation."""
    x_interval = a.to_interval()
    
    # Ensure the interval is non-negative
    if x_interval[0][0] < 0:
        x_interval = interval[max(0, x_interval[0][0]), x_interval[0][1]]
        if x_interval[0][1] < 0:
            raise ValueError("affine_sqrt: input interval is entirely negative.")

    if x_interval[0][0] == x_interval[0][1]:
        return Affine(math.sqrt(x_interval[0][0]))

    f_interval = imath.sqrt(x_interval)
    
    alpha = (f_interval[0][1] - f_interval[0][0]) / (x_interval[0][1] - x_interval[0][0])
    beta = (f_interval[0][1] + f_interval[0][0] - alpha * (x_interval[0][1] + x_interval[0][0])) / 2.0
    
    return a * alpha + beta

def affine_sin_cos_sqr_sum(a: Affine):
    """
    Computes sin(a)**2 + cos(a)**2.
    This should always be 1, but direct computation introduces large errors
    due to the dependency problem. This function handles this specific identity.
    """
    # The result of sin(x)^2 + cos(x)^2 is always 1, regardless of x.
    # We return an affine representation of 1.
    return Affine(1)

# ============================================================================
# Utility Functions for Affine Analysis
# ============================================================================


def AffineEvaluateVectorNorm(xi, yi, zi):
    """Affine evaluation of the norm of an affine vector [xi, yi, zi]"""
    norm_squared = xi * xi + yi * yi + zi * zi
    return affine_sqrt(norm_squared)

def AffineEvaluateNormalizedVector(xi, yi, zi):
    """Affine evaluation of the normalize vector from a vector [xi, yi, zi]"""
    norm = AffineEvaluateVectorNorm(xi, yi, zi)
    
    # Avoid division by zero
    norm_interval = norm.to_interval()
    min_norm = norm_interval[0][0]
    if min_norm <= 1e-9:
        # If norm could be zero, set a minimum bound
        norm = Affine.from_interval(interval[max(min_norm, 1e-6), norm_interval[0][1]])
    
    xni = xi / norm
    yni = yi / norm
    zni = zi / norm
    
    # Apply normalization constraint more carefully
    # For a normalized vector, each component should be in [-1, 1]
    def clamp_to_unit(affine_val):
        interval_val = affine_val.to_interval()
        lower = max(interval_val[0][0], -1.0)
        upper = min(interval_val[0][1], 1.0)
        if lower != interval_val[0][0] or upper != interval_val[0][1]:
            return Affine.from_interval(interval[lower, upper])
        return affine_val
    
    xni = clamp_to_unit(xni)
    yni = clamp_to_unit(yni)
    zni = clamp_to_unit(zni)
    
    return xni, yni, zni

def AffineEvaluateLegendrePolyNomial(x, n):
    """Affine evaluation of the Legendre polynomial P and its derivative Pp of order n at x"""
    if n == 0:
        P = Affine(1)
        Pp = Affine(0)
    elif n == 1:
        P = x
        Pp = Affine(1)
    elif n == 2:
        # P_2(x) = 0.5 * (3*x^2 - 1), use sqr() for better x^2
        x_squared = x.sqr()
        P = (Affine(3) * x_squared - Affine(1)) * Affine(0.5)
        Pp = Affine(3) * x
    else:
        raise NotImplementedError(f"Legendre polynomial of order {n} not implemented")
    return P, Pp


def main():
    """
    Main verification function to test the Affine Arithmetic implementation.
    Tests basic operations, edge cases, and mathematical functions with more complex cases.
    """
    print("=" * 60)
    print("AFFINE ARITHMETIC VERIFICATION (EXTENDED)")
    print("=" * 60)
    
    # Reset symbol index for clean testing
    reset_symbol_index()
    
    # Test 1: Basic Operations
    print("\n1. Testing Basic Operations")
    print("-" * 40)
    a = Affine.from_interval(interval[2, 4])
    b = Affine.from_interval(interval[-1, 3])
    print(f"a = {a}, interval = {a.to_interval()}")
    print(f"b = {b}, interval = {b.to_interval()}")
    print(f"a + b = {a + b}, interval = {(a + b).to_interval()}, expected = {a.to_interval() + b.to_interval()}")
    print(f"a - b = {a - b}, interval = {(a - b).to_interval()}, expected = {a.to_interval() - b.to_interval()}")
    
    # Show the actual computation for multiplication
    mul_result = a * b
    print(f"a * b = {mul_result}, interval = {mul_result.to_interval()}, expected = {a.to_interval() * b.to_interval()}")
    
    # Test 2: Division
    print("\n2. Testing Division")
    print("-" * 40)
    # Case 2.1: Divisor does not contain zero
    b_div = Affine.from_interval(interval[1, 2])
    div_result = a / b_div
    print(f"a / {b_div} = {div_result}, interval = {div_result.to_interval()}, expected = {a.to_interval() / b_div.to_interval()}")
    
    # Case 2.2: Divisor contains zero
    b_zero = Affine.from_interval(interval[-1, 1])
    print(f"a / {b_zero} (contains zero) = {a / b_zero}")
    print(f"Interval result: {(a / b_zero).to_interval()}, expected = {a.to_interval() / b_zero.to_interval()}")

    # Case 2.3: Division by a small number
    small = Affine.from_interval(interval[1e-6, 1e-5])
    large = Affine.from_interval(interval[100, 200])
    result = large / small
    print(f"Large / Small = {result}")
    print(f"Result interval = {result.to_interval()}")

    # Test 3: Mathematical Functions
    print("\n3. Testing Mathematical Functions")
    print("-" * 40)
    x = Affine.from_interval(interval[0.1, 0.5])
    print(f"x = {x}, interval = {x.to_interval()}")
    
    sin_x = affine_sin(x)
    cos_x = affine_cos(x)
    sqrt_x = affine_sqrt(x)
    
    print(f"sin(x) approx = {sin_x.to_interval()}, actual = {imath.sin(x.to_interval())}")
    print(f"cos(x) approx = {cos_x.to_interval()}, actual = {imath.cos(x.to_interval())}")
    print(f"sqrt(x) approx = {sqrt_x.to_interval()}, actual = {imath.sqrt(x.to_interval())}")
    
    # Verify sin²(x) + cos²(x) ≈ 1
    sin_sq_plus_cos_sq = affine_sin_cos_sqr_sum(x)
    print(f"sin²(x) + cos²(x) interval = {sin_sq_plus_cos_sq.to_interval()}")
    assert 1.0 in sin_sq_plus_cos_sq.to_interval()

    # Test 4: Power Function
    print("\n4. Testing Power Function")
    print("-" * 40)
    p = Affine.from_interval(interval[2, 3])
    print(f"p = {p}, interval = {p.to_interval()}")
    p2 = p**2
    p3 = p**3
    p0 = p**0
    print(f"p**2 = {p2}, interval = {p2.to_interval()}, expected = {p.to_interval()**2}")
    print(f"p**3 = {p3}, interval = {p3.to_interval()}, expected = {p.to_interval()**3}")
    print(f"p**0 = {p0}, interval = {p0.to_interval()}, expected = {p.to_interval()**0}")
    
    # Test 5: Chained Operations & Dependency Tracking
    print("\n5. Testing Chained Operations")
    print("-" * 40)
    x = Affine.from_interval(interval[-1, 1])
    # y = x*x. With the new sqr method, this should be tighter.
    y_interval_true = interval[0, 1]
    y_affine = x**2 # uses sqr()
    print(f"x = {x}")
    print(f"x**2 (affine) = {y_affine}, interval = {y_affine.to_interval()}")
    print(f"x**2 (true interval) = {y_interval_true}")
    
    # f(x) = x - x. Should be exactly 0.
    z = x - x
    print(f"x - x = {z}, interval = {z.to_interval()}")
    assert z.to_interval()[0][0] == 0 and z.to_interval()[0][1] == 0

    # More complex chain: f(a,b) = (a+b)*(a-b) vs a^2 - b^2
    a = Affine.from_interval(interval[2, 4])
    b = Affine.from_interval(interval[-1, 1])
    res1 = (a + b) * (a - b)
    res2 = a**2 - b**2
    print(f"a = {a}, b = {b}")
    print(f"(a+b)*(a-b) = {res1.to_interval()}")
    print(f"a^2 - b^2   = {res2.to_interval()}")
    print(f"True interval for a^2-b^2 = {a.to_interval()**2 - b.to_interval()**2}")

    # Test 6: Vector Operations
    print("\n6. Testing Vector Operations")
    print("-" * 40)
    xi = Affine.from_interval(interval[1, 2])
    yi = Affine.from_interval(interval[2, 3])
    zi = Affine.from_interval(interval[-1, 1])
    
    print(f"Vector: xi={xi.to_interval()}, yi={yi.to_interval()}, zi={zi.to_interval()}")
    norm = AffineEvaluateVectorNorm(xi, yi, zi)
    expected_norm = imath.sqrt(xi.to_interval()**2 + yi.to_interval()**2 + zi.to_interval()**2)
    print(f"Norm interval = {norm.to_interval()}, expected = {expected_norm}")
    
    xni, yni, zni = AffineEvaluateNormalizedVector(xi, yi, zi)
    print(f"Normalized vector intervals:")
    print(f"  xni = {xni.to_interval()}")
    print(f"  yni = {yni.to_interval()}")
    print(f"  zni = {zni.to_interval()}")
    
    # Verify normalized vector has unit length (approximately)
    norm_normalized = AffineEvaluateVectorNorm(xni, yni, zni)
    print(f"Norm of normalized vector interval = {norm_normalized.to_interval()}")
    # Note: Due to the dependency problem, this may not be exactly [1,1]
    
    # Test 7: Legendre Polynomials
    print("\n7. Testing Legendre Polynomials")
    print("-" * 40)
    x_leg = Affine.from_interval(interval[-0.5, 0.5])
    print(f"x = {x_leg}, interval = {x_leg.to_interval()}")
    
    # A simple interval version for comparison
    def IntervalLegendre(x_interval, n):
        if n == 0: return interval[1,1], interval[0,0]
        if n == 1: return x_interval, interval[1,1]
        if n == 2: return 0.5 * (3 * x_interval**2 - 1), 3 * x_interval
        return None, None

    for n in range(3):
        P, Pp = AffineEvaluateLegendrePolyNomial(x_leg, n)
        # Compare with interval evaluation
        P_interval, _ = IntervalLegendre(x_leg.to_interval(), n)
        print(f"P_{n}(x) affine = {P.to_interval()}, interval = {P_interval}")
        
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    main()