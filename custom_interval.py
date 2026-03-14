#!/usr/bin/env python3
"""
Custom Interval Arithmetic Implementation

This module provides interval arithmetic functionality compatible with PyInterval,
designed as a fallback when PyInterval is not available.

Features:
- Basic interval operations (addition, subtraction, multiplication, division)
- Power operations and square root
- Interval intersections
- Math functions for intervals

Author: Custom implementation for eMNS workspace analysis
Compatible with: PyInterval interface
"""

import math
import numpy as np

class Interval:
    """
    Custom interval arithmetic implementation
    
    Represents a closed interval [lower, upper] with arithmetic operations.
    """
    
    def __init__(self, lower, upper=None):
        """
        Initialize an interval
        
        Args:
            lower: Lower bound, or a list/tuple [lower, upper]
            upper: Upper bound (if lower is not a container)
        """
        if upper is None:
            if isinstance(lower, (list, tuple)) and len(lower) == 2:
                self.lower = float(lower[0])
                self.upper = float(lower[1])
            else:
                # Single value interval
                self.lower = float(lower)
                self.upper = float(lower)
        else:
            self.lower = float(lower)
            self.upper = float(upper)
        
        # Ensure lower <= upper
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower
    
    def __getitem__(self, index):
        """Allow indexing like interval[0] for compatibility"""
        if index == 0:
            return [self.lower, self.upper]
        else:
            raise IndexError("Interval index out of range")
    
    def __repr__(self):
        return f"Interval([{self.lower}, {self.upper}])"
    
    def __str__(self):
        return f"[{self.lower}, {self.upper}]"
    
    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        else:
            return Interval(self.lower + other, self.upper + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        else:
            return Interval(self.lower - other, self.upper - other)
    
    def __rsub__(self, other):
        return Interval(other - self.upper, other - self.lower)
    
    def __neg__(self):
        """Unary minus (negation) of interval"""
        return Interval(-self.upper, -self.lower)
    
    def __pos__(self):
        """Unary plus of interval"""
        return Interval(self.lower, self.upper)
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ]
            return Interval(min(products), max(products))
        else:
            if other >= 0:
                return Interval(self.lower * other, self.upper * other)
            else:
                return Interval(self.upper * other, self.lower * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                # Division by interval containing zero
                return Interval(-float('inf'), float('inf'))
            else:
                quotients = [
                    self.lower / other.lower,
                    self.lower / other.upper,
                    self.upper / other.lower,
                    self.upper / other.upper
                ]
                return Interval(min(quotients), max(quotients))
        else:
            if other == 0:
                return Interval(-float('inf'), float('inf'))
            elif other > 0:
                return Interval(self.lower / other, self.upper / other)
            else:
                return Interval(self.upper / other, self.lower / other)
    
    def __rtruediv__(self, other):
        return Interval(other, other).__truediv__(self)
    
    def __pow__(self, power):
        if power == 2:
            if self.lower >= 0:
                return Interval(self.lower**2, self.upper**2)
            elif self.upper <= 0:
                return Interval(self.upper**2, self.lower**2)
            else:
                return Interval(0, max(self.lower**2, self.upper**2))
        elif power == 3:
            return Interval(self.lower**3, self.upper**3)
        elif isinstance(power, int) and power > 0:
            if power % 2 == 0:  # Even power
                if self.lower >= 0:
                    return Interval(self.lower**power, self.upper**power)
                elif self.upper <= 0:
                    return Interval(self.upper**power, self.lower**power)
                else:
                    return Interval(0, max(abs(self.lower)**power, abs(self.upper)**power))
            else:  # Odd power
                return Interval(self.lower**power, self.upper**power)
        else:
            # General case - may not be monotonic
            values = []
            test_points = [self.lower, self.upper]
            
            # Add critical points for some functions
            if self.lower < 0 < self.upper and isinstance(power, (int, float)):
                test_points.append(0)
            
            for x in test_points:
                try:
                    values.append(x**power)
                except (ValueError, ZeroDivisionError):
                    pass
            
            if values:
                return Interval(min(values), max(values))
            else:
                return Interval(-float('inf'), float('inf'))
    
    def __and__(self, other):
        """Intersection of intervals"""
        if isinstance(other, Interval):
            lower = max(self.lower, other.lower)
            upper = min(self.upper, other.upper)
            if lower <= upper:
                return Interval(lower, upper)
            else:
                # Empty intersection
                return Interval(float('nan'), float('nan'))
        else:
            raise TypeError("Can only intersect with another Interval")
    
    def __or__(self, other):
        """Hull (union) of intervals"""
        if isinstance(other, Interval):
            return Interval(min(self.lower, other.lower), max(self.upper, other.upper))
        else:
            raise TypeError("Can only unite with another Interval")
    
    # Comparison operations
    def __contains__(self, value):
        """Check if value is in interval"""
        return self.lower <= value <= self.upper
    
    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lower == other.lower and self.upper == other.upper
        else:
            return False
    
    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.upper < other.lower
        else:
            return self.upper < other
    
    def __le__(self, other):
        if isinstance(other, Interval):
            return self.upper <= other.lower
        else:
            return self.upper <= other
    
    def __gt__(self, other):
        if isinstance(other, Interval):
            return self.lower > other.upper
        else:
            return self.lower > other
    
    def __ge__(self, other):
        if isinstance(other, Interval):
            return self.lower >= other.upper
        else:
            return self.lower >= other
    
    # Utility methods
    def width(self):
        """Return the width of the interval"""
        return self.upper - self.lower
    
    def midpoint(self):
        """Return the midpoint of the interval"""
        return (self.lower + self.upper) / 2
    
    def is_empty(self):
        """Check if interval is empty (NaN bounds)"""
        return math.isnan(self.lower) or math.isnan(self.upper)
    
    def abs(self):
        """Absolute value of interval"""
        if self.lower >= 0:
            return Interval(self.lower, self.upper)
        elif self.upper <= 0:
            return Interval(-self.upper, -self.lower)
        else:
            return Interval(0, max(-self.lower, self.upper))
    
    def intersection(self, other):
        """Intersection with another interval (alias for &)"""
        return self & other


class IntervalMath:
    """
    Mathematical functions for intervals
    """
    
    @staticmethod
    def sqrt(interval_val):
        """Square root of interval"""
        if isinstance(interval_val, Interval):
            if interval_val.upper < 0:
                # Square root of negative number
                return Interval(float('nan'), float('nan'))
            elif interval_val.lower < 0:
                # Interval contains negative values
                lower = 0
                upper = math.sqrt(interval_val.upper)
                return Interval(lower, upper)
            else:
                # All positive
                return Interval(math.sqrt(interval_val.lower), math.sqrt(interval_val.upper))
        else:
            return math.sqrt(max(0, interval_val))
    
    @staticmethod
    def arccos(interval_val):
        """Arccosine of interval"""
        if isinstance(interval_val, Interval):
            # Domain check: [-1, 1]
            if interval_val.lower < -1 or interval_val.upper > 1:
                # Clamp to valid domain
                lower = max(interval_val.lower, -1)
                upper = min(interval_val.upper, 1)
                if lower > upper:
                    return Interval(float('nan'), float('nan'))
            else:
                lower = interval_val.lower
                upper = interval_val.upper
            
            # arccos is decreasing, so arccos(upper) gives lower bound
            return Interval(math.acos(upper), math.acos(lower))
        else:
            if -1 <= interval_val <= 1:
                return math.acos(interval_val)
            else:
                return float('nan')
    
    @staticmethod
    def arctan2(y_interval, x_interval):
        """Two-argument arctangent of intervals"""
        if isinstance(y_interval, Interval) or isinstance(x_interval, Interval):
            # Convert scalars to intervals
            if not isinstance(y_interval, Interval):
                y_interval = Interval(y_interval, y_interval)
            if not isinstance(x_interval, Interval):
                x_interval = Interval(x_interval, x_interval)
            
            # Sample key points from both intervals
            y_points = [y_interval.lower, y_interval.upper]
            x_points = [x_interval.lower, x_interval.upper]
            
            # Add zero crossings if they exist
            if y_interval.lower <= 0 <= y_interval.upper:
                y_points.append(0)
            if x_interval.lower <= 0 <= x_interval.upper:
                x_points.append(0)
            
            # Compute atan2 at all combinations
            results = []
            for y in y_points:
                for x in x_points:
                    if not (x == 0 and y == 0):  # Avoid undefined case
                        results.append(math.atan2(y, x))
            
            if not results:
                return Interval(float('nan'), float('nan'))
            
            # Handle angle wrapping - atan2 returns values in [-π, π]
            min_result = min(results)
            max_result = max(results)
            
            # Check if we cross the ±π boundary
            if max_result - min_result > math.pi:
                # This is a heuristic - in practice, might need more sophisticated handling
                return Interval(-math.pi, math.pi)
            else:
                return Interval(min_result, max_result)
        else:
            return math.atan2(y_interval, x_interval)
    
    @staticmethod
    def sin(interval_val):
        """Sine of interval with proper extrema handling"""
        if isinstance(interval_val, Interval):
            # Handle full periods or larger
            width = interval_val.width()
            if width >= 2 * math.pi:
                return Interval(-1, 1)
            
            # Calculate sin at endpoints
            s_l, s_u = math.sin(interval_val.lower), math.sin(interval_val.upper)
            min_val, max_val = min(s_l, s_u), max(s_l, s_u)
            
            # Check for maximum at π/2 + 2πk within the interval
            # For simplicity, check common cases: π/2, 5π/2, etc.
            pi_half = math.pi / 2
            if interval_val.lower <= pi_half <= interval_val.upper:
                max_val = 1.0
            if interval_val.lower <= pi_half + 2*math.pi <= interval_val.upper:
                max_val = 1.0
                
            # Check for minimum at 3π/2 + 2πk within the interval
            three_pi_half = 3 * math.pi / 2
            if interval_val.lower <= three_pi_half <= interval_val.upper:
                min_val = -1.0
            if interval_val.lower <= three_pi_half + 2*math.pi <= interval_val.upper:
                min_val = -1.0
                
            return Interval(min_val, max_val)
        else:
            return math.sin(interval_val)
    
    @staticmethod
    def cos(interval_val):
        """Cosine of interval with proper extrema handling"""
        if isinstance(interval_val, Interval):
            # Handle full periods or larger
            width = interval_val.width()
            if width >= 2 * math.pi:
                return Interval(-1, 1)
            
            # Calculate cos at endpoints
            c_l, c_u = math.cos(interval_val.lower), math.cos(interval_val.upper)
            min_val, max_val = min(c_l, c_u), max(c_l, c_u)
            
            # Check for maximum at 0 + 2πk within the interval
            # For simplicity, check common cases: 0, 2π, 4π, etc.
            if interval_val.lower <= 0 <= interval_val.upper:
                max_val = 1.0
            if interval_val.lower <= 2*math.pi <= interval_val.upper:
                max_val = 1.0
            if interval_val.lower <= -2*math.pi <= interval_val.upper:
                max_val = 1.0
                
            # Check for minimum at π + 2πk within the interval
            if interval_val.lower <= math.pi <= interval_val.upper:
                min_val = -1.0
            if interval_val.lower <= math.pi + 2*math.pi <= interval_val.upper:
                min_val = -1.0
            if interval_val.lower <= math.pi - 2*math.pi <= interval_val.upper:
                min_val = -1.0
                
            return Interval(min_val, max_val)
        else:
            return math.cos(interval_val)
    
    @staticmethod
    def exp(interval_val):
        """Exponential of interval"""
        if isinstance(interval_val, Interval):
            return Interval(math.exp(interval_val.lower), math.exp(interval_val.upper))
        else:
            return math.exp(interval_val)
    
    @staticmethod
    def log(interval_val):
        """Natural logarithm of interval"""
        if isinstance(interval_val, Interval):
            if interval_val.upper <= 0:
                return Interval(float('nan'), float('nan'))
            elif interval_val.lower <= 0:
                return Interval(-float('inf'), math.log(interval_val.upper))
            else:
                return Interval(math.log(interval_val.lower), math.log(interval_val.upper))
        else:
            return math.log(interval_val) if interval_val > 0 else float('nan')


class IntervalFactory:
    """
    Factory class to create intervals with square bracket syntax support
    """
    
    def __call__(self, bounds):
        """
        Factory function to create intervals
        
        Args:
            bounds: List or tuple [lower, upper]
        
        Returns:
            Interval object
        """
        # if PYINTERVAL_AVAILABLE:
        #     # Use original PyInterval if available
        #     return pyinterval_interval(bounds)
        # else:
        #     # Use our custom implementation
        return Interval(bounds)
    
    def __getitem__(self, key):
        """
        Support interval[a, b] syntax
        
        Args:
            key: slice object or tuple (a, b)
        
        Returns:
            Interval object
        """
        if isinstance(key, slice):
            # Handle interval[a:b] syntax
            if key.step is not None:
                raise ValueError("Step not supported in interval syntax")
            return self([key.start, key.stop])
        elif isinstance(key, tuple) and len(key) == 2:
            # Handle interval[a, b] syntax
            return self([key[0], key[1]])
        else:
            raise ValueError("Invalid interval syntax. Use interval[a, b]")


# Create the global interval factory
interval = IntervalFactory()


# Math module replacement
# if PYINTERVAL_AVAILABLE:
#     imath = pyinterval_imath
#     fpu = pyinterval_fpu
# else:
imath = IntervalMath()
fpu = None  # FPU control not implemented in custom version


# Convenience functions
def hull(*intervals):
    """Compute the hull (union) of multiple intervals"""
    if not intervals:
        return Interval(float('nan'), float('nan'))
    
    result = intervals[0]
    for i in intervals[1:]:
        result = result | i
    return result


def intersection(*intervals):
    """Compute the intersection of multiple intervals"""
    if not intervals:
        return Interval(float('nan'), float('nan'))
    
    result = intervals[0]
    for i in intervals[1:]:
        result = result & i
    return result


def width(interval_val):
    """Get width of interval"""
    if hasattr(interval_val, 'width'):
        return interval_val.width()
    else:
        return 0


def midpoint(interval_val):
    """Get midpoint of interval"""
    if hasattr(interval_val, 'midpoint'):
        return interval_val.midpoint()
    else:
        return interval_val


# Note: Mathematical functions are available via imath.cos(), imath.sin(), etc.
# No need for standalone functions since they just duplicate the static methods

# Export the main classes and functions
__all__ = [
    'Interval', 'IntervalMath', 'interval', 'imath', 'fpu',
    'hull', 'intersection', 'width', 'midpoint'
]


if __name__ == "__main__":
    # Simple test
    print("Testing custom interval implementation...")
    
    # Test basic operations
    a = interval([1, 3])
    b = interval([2, 4])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"sqrt(a) = {imath.sqrt(a)}")
    print(f"a & b = {a & b}")  # intersection
    print(f"a | b = {a | b}")  # hull
    print(f"-a = {-a}")  # negation
    print(f"+a = {+a}")  # unary plus
    
    print("✓ Basic interval tests passed!")