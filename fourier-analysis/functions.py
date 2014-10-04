# This Sage pane has all the functions that we'll use in the specific computation code.
# It doesn't do any computation, but rather only defines functions.  You should run it
# before everything else, and should run it after changing any of its functions.

import cmath
from itertools import izip
import math
import numpy as np
import pickle
from scipy import integrate

# must provide one of the 'value_count' or 'step_size' keyword arguments.
# the 'start' and 'end' arguments must be ones for which addition,
# subtraction, multiplication-by-integer and division-by-integer are
# defined.
def linterp (start, end, **keywords):
    r"""
    Returns a generator function for linearly interpolating between the `start` and `end` values.
    
    Must provide exactly one of the following keyword arguments:
    - `value_count` : The returned generator will produce exactly value_count values (value_count must be at least 2).
    - `step_size`   : The absolute value of the increment between generated values.

    If `value_count` is specified, then the values specified for `start` and `end` can be anything that behaves as
    a vector (specifically, has defined addition, subtraction, and scalar multiplication operations).  Otherwise,
    the values specified for `start` and `end` must be scalar values which have an ordering relation (__gt__) defined.
    """
    
    # logical XOR
    assert ('value_count' in keywords) != ('step_size' in keywords), "must specify exactly one of 'value_count' or 'step_size'"
    if 'value_count' in keywords:
        value_count = keywords['value_count']
        assert value_count >= 2
        step_count = value_count - 1
        for i in range(value_count-1):
            yield start + (end - start) * i / step_count
        yield end # yield end exactly
    else:
        step_size = keywords['step_size']
        assert step_size > 0
        value = start
        if end > start:
            while value <= end:
                yield value
                value += step_size
        else:
            while value >= end:
                yield value
                value -= step_size

def PickleThis (obj, filename):
    pickle.dump(obj, open(filename, "wb"))

def UnpickleThis (filename):
    return pickle.load(open(filename, "rb"))

def PrintNiceList (L):
    for x in L:
        print x

def ComplexVector (*components):
    return np.ndarray(shape=(len(components),), dtype=complex, buffer=np.array([complex(c) for c in components]))

def ComplexConjugate (complex_vector):
    return ComplexVector(*[z.conjugate() for z in complex_vector])

def RealPartOfComplexVector (complex_vector):
    return ComplexVector(*[z.real for z in complex_vector])

def ImagPartOfComplexVector (complex_vector):
    return ComplexVector(*[z.imag for z in complex_vector])

def NormSquared (V):
    if isinstance(V,list):
        return sum(NormSquared(v) for v in V)
    else: # Assume it is a complex vector.
        return V.dot(ComplexConjugate(V)).real

def Norm (V):
    return math.sqrt(NormSquared(V))

def ListDot (X, Y):
    return sum(x*y for (x,y) in izip(X,Y))

def SampledRiemannSum (samples, dt):
    return sum(sample*dt for sample in samples)

def IntegrateComplexScalarFunction (func, a, b):
    # NOTE: this computes func(t) twice because it integrates twice.
    def func_real (t):
        return func(t).real
    def func_imag (t):
        return func(t).imag
    return complex(integrate.quad(func_real, a, b)[0], integrate.quad(func_imag, a, b)[0])

def IntegrateComplexVectorFunction (func, a, b):
    n = len(func(0)) # This is the dimension of the vector that func produces
    # NOTE: this computes func(t) more than once.
    def ComponentFunctionForIndex (i):
        def ComponentFunction (t):
            return func(t)[i]
        return ComponentFunction
    return ComplexVector(*[IntegrateComplexScalarFunction(ComponentFunctionForIndex(i), a, b) for i in range(n)])

# samples must be a list which represents a function f(t) which is sampled
# over the t values given in sample_times.  requested_mode is the 
# integer-valued mode of the Fourier coefficient that will be computed.  In
# particular, this computes the discretized integral
#   \int_{0}^{period} f(t) exp(-i*omega*requested_mode*t) dt,
# where omega = 2*pi/period.
def FourierCoefficient (sample_times, samples, period, requested_mode):
    assert len(sample_times) == len(samples), "sample_times and samples must be of the same length"
    assert period > 0, "period must be positive"
    # TODO: somehow check that requested_mode is an integer
    omega = float(2*math.pi/period)
    n = len(samples)
    # The discrete integral is just a finite Riemann sum, so dt is really just a width.
    dts = [sample_times[i+1] - sample_times[i] for i in range(n-1)] + [period - sample_times[-1]]
    exponent_factor = complex(-1j*omega*requested_mode)
    return sum(f_of_t*cmath.exp(exponent_factor*t)*dt for (t,f_of_t,dt) in izip(sample_times,samples,dts)) / period

def FourierExpansion (sample_times, samples, period, requested_modes):
    return [FourierCoefficient(sample_times, samples, period, requested_mode) for requested_mode in requested_modes]

def FourierSum (modes, coefficients, period, t):
    #print "computing FourierSum for thingy with t = {0}".format(t)
    omega = float(2*math.pi/period)
    exponent_factor = complex(1j*omega)
    return sum(coefficient*cmath.exp(exponent_factor*mode*t) for (mode,coefficient) in izip(modes,coefficients))

# This is the negative gradient of the potential energy function.
def Force (V):
    assert len(V) == 3, "V must be a 3-vector"
    x = V[0]
    y = V[1]
    z = V[2]
    alpha = 1
    r_squared = x**2 + y**2
    mu = r_squared**2 + 0.0625*z**2
    return ComplexVector(-alpha*mu^(-3/2)*r_squared*2*x, \
                         -alpha*mu^(-3/2)*r_squared*2*y, \
                         -.0625*alpha*mu^(-3/2)*z)

def ForceCoefficients (position_modes, position_coefficients, period, force_modes):
    sample_times = [t for t in linterp(float(0), float(period), value_count=1000)]
    force_field_samples = [FourierSum(position_modes, position_coefficients, period, t) for t in sample_times]
    
    #def ForceField (t):
    #    position_at_time_t = FourierSum(position_modes, position_coefficients, period, t)
    #    return Force(position_at_time_t)
    
    omega = float(2*math.pi/period)
    def SampledIntegrandOfMode (mode):
        exponent_factor = complex(-1j*omega*mode)
        integrand_samples = [force_field_sample*cmath.exp(exponent_factor*t) \
                             for (t,force_field_sample) in izip(sample_times,force_field_samples)]
        return integrand_samples

    #def IntegrandOfMode (mode):
    #    exponent_factor = complex(-1j*omega*mode)
    #    def Integrand (t):
    #        return ForceField(t)*exp(exponent_factor*t)
    #    return Integrand
        
    force_coefficients = []
    for mode in force_modes:
        #print "computing force coefficient for mode {0}".format(mode)
        #force_coefficient = IntegrateComplexVectorFunction(IntegrandOfMode(mode), 0, period)/period 
        integrand_samples = SampledIntegrandOfMode(mode)
        dt = period / len(integrand_samples)
        force_coefficient = SampledRiemannSum(integrand_samples, dt) / period
        #print "    ... force_coefficient = {0}".format(force_coefficient)
        force_coefficients.append(force_coefficient)
    #force_coefficients = [IntegrateComplexVectorFunction(IntegrandOfMode(mode), 0, period)/period for mode in force_modes]
    return force_coefficients

def DroppedZCoordinate (complex_3_vector):
    return ComplexVector(complex_3_vector[0], complex_3_vector[1], 0)

def Line2dFromComplex3Vectors (complex_3_vectors, *args, **kwargs):
    return line2d(((v[0].real, v[1].real) for v in complex_3_vectors), *args, **kwargs)

def FourierSpaceMetric (mode, period):
    omega = float(2*math.pi/period)
    return mode**2 * omega**2

def Update (modes, coefficients, period, step_size):
    ratio = float(step_size) / period
    force_coefficients = ForceCoefficients(modes, coefficients, period, modes)
    updated_coefficients = [  coefficient \
                            - ratio*(DroppedZCoordinate(coefficient) \
                            + FourierSpaceMetric(mode, period)*force_coefficient) \
                            for (mode,coefficient,force_coefficient) in izip(modes,coefficients,force_coefficients)]
    return updated_coefficients

# This function starts the updates using the last element of the updates list,
# and appends the updated coefficients to the updates list.  This way, it's not
# replacing any data, only adding to it.  So in theory, you can interrupt the
# worksheet (Esc key) and the updates it has computed so far will be preserved,
# and you can continue again in the future.
def RunUpdates (modes, period, updates, step_count, step_size):
    assert len(updates) > 0
    for i in range(step_count):
        # Let coefficients be the last element of the updates list (the most recent one)
        coefficients = updates[-1]
        print "computing update {0} out of {1} ({2} updates stored so far)".format(i+1, step_count, len(updates))
        # Compute the result of the gradient descent update.
        new_coefficients = Update(modes, coefficients, period, step_size)
        # Append the result to the updates list.
        updates.append(new_coefficients)
        # Compute and print the distance between the most recent and previous coefficients.
        distance = Norm([c-n for (c,n) in izip(coefficients,new_coefficients)])
        print "distance from previous iteration value to next: {0}".format(distance)
        
        ## If there are at least 3 updates, call them U_0, U_1, and U_2 in order from oldest to newest,
        ## then the angle between U_1-U_0 and U_2-U_1 can be computed and used as an informative
        ## indication of if the update step is fluctuating in direction or not.
        #if len(updates) >= 3:
        #    U_0 = updates[-3] # Third to last element
        #    U_1 = updates[-2] # Second to last element
        #    U_2 = updates[-1] # Last element
        #    old_direction = [u_1-u_0 for (u_0,u_1) in izip(U_0,U_1)]
        #    new_direction = [u_2-u_1 for (u_1,u_2) in izip(U_1,U_2)]
        #    old_direction_norm_squared = NormSquared(old_direction)
        #    new_direction_norm_squared = NormSquared(new_direction)
        #    # Only bother if the denominator is not too small.
        #    if old_direction_norm_squared > 1.0e-20 and new_direction_norm_squared > 1.0e-20:
        #        num = ListDot(old_direction,new_direction)
        #        denom = math.sqrt(old_direction_norm_squared*new_direction_norm_squared)
        #        angle_in_degrees = float(arccos(num/denom)*180.0/math.pi)
        #        print "angle from previous update direction to new update direction: {0}".format(angle_in_degrees)