import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

tau = 2.0*math.pi

def cubic_bezier_tensor_j0_slice (t):
    # indexed as bezier_j0[b,d]
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier_j0 = np.ndarray((2,2), dtype=float)
    s = 1.0 - t
    bezier_j0[0,0] =  s**2 * (s + 3*t)
    bezier_j0[0,1] =  s**2 * t
    bezier_j0[1,0] =  t**2 * (3*s + t)
    bezier_j0[1,1] = -t**2 * s
    return bezier_j0

def cubic_bezier_tensor_j1_slice (t):
    # indexed as bezier_j1[b,d]
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier_j1 = np.ndarray((2,2), dtype=float)
    s = 1.0 - t
    bezier_j1[0,0] = -6*s*t
    bezier_j1[0,1] =  s**2 - 2*s*t
    bezier_j1[1,0] =  6*s*t
    bezier_j1[1,1] =  t**2 - 2*s*t
    return bezier_j1

def cubic_bezier_tensor (time_tensor):
    # indexed as bezier[i,b,d]
    # i is the interolation time index
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier = np.ndarray((len(time_tensor),2,2), dtype=float)
    for i,t in enumerate(time_tensor):
        bezier[i,:,:] = cubic_bezier_tensor_j0_slice(t)
    return bezier

def cubic_bezier_derivative_tensor (time_tensor):
    # indexed as bezier_derivative[i,b,d]
    # i is the interolation time index
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier_derivative = np.ndarray((len(time_tensor),2,2), dtype=float)
    for i,t in enumerate(time_tensor):
        # s = 1.0 - t
        # bezier_derivative[i,0,0] = -6*s*t
        # bezier_derivative[i,0,1] =  s**2 - 2*s*t
        # bezier_derivative[i,1,0] =  6*s*t
        # bezier_derivative[i,1,1] =  t**2 - 2*s*t
        bezier_derivative[i,:,:] = cubic_bezier_tensor_j1_slice(t)
    return bezier_derivative

def cubic_bezier_jet_tensor_slice (t):
    # indexed as bezier_jet[c,b,d]
    # c is the derivative index of the output (cth derivative)
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier_jet = np.ndarray((2,2,2), dtype=float)
    bezier_jet[0,:,:] = cubic_bezier_tensor_j0_slice(t)
    bezier_jet[1,:,:] = cubic_bezier_tensor_j1_slice(t)
    return bezier_jet

def cubic_bezier_jet_tensor (time_tensor):
    # indexed as bezier_jet[c,i,b,d]
    # c is the derivative index of the output (cth derivative)
    # i is the interolation time index
    # b is the boundary index (0:initial point, 1:terminal point)
    # d is the derivative index (dth derivative)
    bezier_jet = np.ndarray((2,len(time_tensor),2,2), dtype=float)
    bezier_jet[0,:,:,:] = cubic_bezier_tensor(time_tensor)
    bezier_jet[1,:,:,:] = cubic_bezier_derivative_tensor(time_tensor)
    return bezier_jet

def cycle_tensor (cycle_size, cyclic_advancement_count):
    # This really should be a sparse matrix.
    cycle = np.zeros((cycle_size,cycle_size), dtype=float)
    for c in xrange(cycle_size):
        cycle[c,(c+cyclic_advancement_count)%cycle_size] = 1.0
    return cycle

def adjacent_cyclic_pair_tensor (cycle_size):
    # Indexed as acp[s,t,b], where
    # s is the output interpolation time index in [0,cycle_size)
    # t is the input interpolation time index in [0,cycle_size)
    # b is the boundary index (0:initial point, 1:terminal point)
    acp = np.ndarray((cycle_size,cycle_size,2), dtype=float)
    acp[:,:,0] = np.eye(cycle_size, dtype=float)
    acp[:,:,1] = cycle_tensor(cycle_size, 1)
    return acp

def demo_cubic_bezier ():
    T = 30

    xv_shape = (2,2)
    # xv is indexeed as xv[b,d], where b is the index for the boundary (0:start or 1:end)
    # and d is the differential degree of the jet value (0:position or 1:velocity).
    xv = np.random.randn(*xv_shape)
    # # This tensor does the cubic Bezier interpolation
    # b = np.ndarray((T,2,2), dtype=float)
    # for t in xrange(T):
    #     p = float(t)/(T-1)
    #     q = 1.0 - p
    #     b[t,0,0] =  q**2 * (q + 3*p)
    #     b[t,0,1] =  q**2 * p
    #     b[t,1,0] =  p**2 * (3*q + p)
    #     b[t,1,1] = -p**2 * q

    time = np.linspace(0.0, 1.0, T)
    bezier = cubic_bezier_tensor(time)
    bezier_curve = np.einsum('tbd,bd', bezier, xv)

    plt.figure(1)
    plt.plot([0.0,0.5], [xv[0,0],xv[0,0]+0.5*xv[0,1]])
    plt.plot([0.5,1.0], [xv[1,0]-0.5*xv[1,1],xv[1,0]])
    plt.plot(time, bezier_curve)
    plt.show()

def demo_cyclic_cubic_bezier ():
    T = 30
    N = 10
    time = np.linspace(0.0, 1.0, T+1)[:-1]
    xv_shape = (2,N)
    xv = np.array([[math.cos(tau/N*float(n)) for n in xrange(N)], [-tau/N*math.sin(tau/N*float(n)) for n in xrange(N)]])

    acp = adjacent_cyclic_pair_tensor(N)
    # b indexes 0:initial point or 1:terminal point
    # d indexes 0:position or 1:velocity
    # m and n index which point in the cycle
    acp_xv = np.einsum('nmb,dm->bdn', acp, xv)

    bezier = cubic_bezier_tensor(time)
    bezier_derivative = cubic_bezier_derivative_tensor(time)
    bezier_jet = cubic_bezier_jet_tensor(time)
    # The reshaping is to get the whole interpolated sequence into a single index.
    # If you want each segment interpolation to be separately indexed, don't reshape here.
    cyclic_bezier_curve = np.einsum('tbd,bdn->nt', bezier, acp_xv).reshape(T*N)
    cyclic_bezier_derivative_curve = np.einsum('tbd,bdn->nt', bezier_derivative, acp_xv).reshape(T*N)

    approximate_actual_derivative_curve = np.diff(cyclic_bezier_curve)/time[1]
    print np.max(np.abs(cyclic_bezier_derivative_curve[:-1] - approximate_actual_derivative_curve))

    cyclic_bezier_jet_curve = np.einsum('ctbd,bdn->cnt', bezier_jet, acp_xv).reshape(2,T*N)
    print np.max(np.abs(cyclic_bezier_curve - cyclic_bezier_jet_curve[0,:]))
    print np.max(np.abs(cyclic_bezier_derivative_curve - cyclic_bezier_jet_curve[1,:]))

    fig,axes = plt.subplots(2, 1, figsize=(10,5))

    a = axes[0]
    a.plot(range(0, N*T, T), xv[0,:], color='blue')
    a.plot([N*T-T, N*T], [xv[0,-1], xv[0,0]], color='blue')
    a.plot(cyclic_bezier_curve, color='green')

    a = axes[1]
    a.plot(approximate_actual_derivative_curve, color='blue')
    a.plot(cyclic_bezier_derivative_curve, color='green')

    plt.show()

def demo_continuous_cubic_bezier ():
    N = 10
    xv_shape = (2,N)
    omega = tau/N
    xv = np.array([[math.cos(omega*float(n)) for n in xrange(N)], [-omega*math.sin(omega*float(n)) for n in xrange(N)]])

    acp = adjacent_cyclic_pair_tensor(N)
    # b indexes 0:initial point or 1:terminal point
    # d indexes 0:position or 1:velocity
    # m and n index which point in the cycle
    acp_xv = np.einsum('nmb,dm->bdn', acp, xv)

    def n_for_cyclic_time (t):
        return int(math.floor(t*N))%N

    def eval_cyclic_x (t):
        n = n_for_cyclic_time(t)
        t *= N
        t = (t - math.floor(t))
        assert 0 <= t < 1
        bezier_slice = cubic_bezier_tensor_j0_slice(t)
        return np.einsum('bd,bd', bezier_slice, acp_xv[:,:,n])

    def n_for_clamped_time (t):
        return min(max(int(math.floor(t*N)), 0), N-1)

    def eval_clamped_x (t):
        n = n_for_clamped_time(t)
        t *= N
        if 0 <= t < N:
            t = (t - math.floor(t))
        elif N <= t:
            t = t - (N-1)
        bezier_slice = cubic_bezier_tensor_j0_slice(t)
        return np.einsum('bd,bd', bezier_slice, acp_xv[:,:,n])

    T = 300
    time = np.linspace(-2.0, 3.0, T+1)[:-1]
    cyclic_curve = np.array([eval_cyclic_x(t) for t in time])
    clamped_curve = np.array([eval_clamped_x(t) for t in time])

    fig,axes = plt.subplots(2, 1, figsize=(6,6))
    axes[0].plot(time, cyclic_curve)
    axes[1].plot(time, clamped_curve)
    axes[1].set_ylim(-1.0, 1.0)
    plt.show()

if __name__ == '__main__':
    # print cycle_tensor(10, 0)
    # print cycle_tensor(10, 1)
    # print cycle_tensor(10, 2)
    # demo_cubic_bezier()
    # demo_cyclic_cubic_bezier()
    demo_continuous_cubic_bezier()


