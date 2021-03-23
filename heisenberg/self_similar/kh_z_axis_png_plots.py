import heisenberg.self_similar.kh
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.interpolate
import typing
import vorpy
import vorpy.realfunction.bezier
import vorpy.realfunction.piecewiselinear

def plot_J_equal_zero_extrapolated_trajectory (p_y_initial:float) -> None:
    x_initial = 1.0
    y_initial = 0.0
    z_initial = 0.0
    p_x_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = heisenberg.self_similar.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    p_z_initial = qp_initial[1,2]

    pickle_file_p = pathlib.Path(f'SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_dilation.temp/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = heisenberg.self_similar.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=15.0,
        solution_sheet=0,
    )

    #
    # Try to construct the t < 0 portion of the solution.
    #

    # Solve for the smallest P > 0 such that z(P) == 0.

    t_v = results.t_v
    qp_v = results.y_t
    z_v = qp_v[:,0,2]

    #print(z_v)

    z_zero_index_pair_v, z_zero_orientation_v, z_zero_t_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v)
    #print(np.vstack((z_zero_index_pair_v.T, z_zero_orientation_v, z_zero_t_v)))
    assert len(z_zero_t_v) >= 2
    assert z_zero_t_v[0] == t_v[0]

    P_bound_index = z_zero_index_pair_v[1,1]
    P = z_zero_t_v[1]
    assert P <= t_v[P_bound_index]

    # TODO: Use Bezier interpolation instead
    qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
    qp_P = qp_interpolator(P)

    segment_t_v = -t_v[:P_bound_index+1]
    segment_qp_v = np.copy(qp_v[:P_bound_index+1,...])

    # Reverse these so that time goes forward.
    segment_t_v[...] = segment_t_v[::-1]
    segment_qp_v[...] = segment_qp_v[::-1,...]

    # Apply transformations to segment_qp_v (flip y, z, p_y, p_z, then flip p_x, p_y, p_z,
    # which is equivalent to flipping y, z, p_x).
    segment_qp_v[:,0,1] *= -1
    segment_qp_v[:,0,2] *= -1
    segment_qp_v[:,1,0] *= -1

    # TODO: Use Bezier interpolation instead
    segment_qp_interpolator = scipy.interpolate.interp1d(segment_t_v, segment_qp_v, axis=0)
    segment_qp_minus_P = segment_qp_interpolator(-P)

    angle_v = np.arctan2(qp_P[:,1], qp_P[:,0]) - np.arctan2(segment_qp_minus_P[:,1], segment_qp_minus_P[:,0])
    for i in range(len(angle_v)):
        while angle_v[i] < 0:
            angle_v[i] += 2*np.pi

    print(f'angle_v = {angle_v}')
    if np.max(angle_v) - np.min(angle_v) < 1.0e-6:
        print(f'angles matched as expected')
    else:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!\n! ANGLES DID NOT MATCH !\n!!!!!!!!!!!!!!!!!!!!!!!!')

    angle = np.mean(angle_v)

    R = np.array([
        [ np.cos(-angle), np.sin(-angle), 0.0],
        [-np.sin(-angle), np.cos(-angle), 0.0],
        [            0.0,            0.0, 1.0],
    ])
    extrapolated_segment_qp_v = np.einsum('ij,tpj->tpi', R, segment_qp_v)
    interpolated_segment_t_v = segment_t_v+2*P
    interpolated_segment_qp_v = qp_interpolator(interpolated_segment_t_v)

    extrapolation_error = np.max(np.abs(extrapolated_segment_qp_v - interpolated_segment_qp_v))

    print(f'extrapolation_error = {extrapolation_error}')
    if extrapolation_error > 5.0e-5:
        print('!!!!!!!!!!!!!!!!!!\n! ERROR EXCEEDED !\n!!!!!!!!!!!!!!!!!!')

    row_count   = 2
    col_count   = 2
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for phase_index in range(2):
        s = 'p_' if phase_index == 1 else ''
        angle = angle_v[phase_index]

        axis = axis_vv[phase_index][0]
        axis.set_title(f'initial ({s}x, {s}y) = {(p_x_initial, p_y_initial)}\n({s}x(t), {s}y(t))\nblue:solution, green:extrapolated\nangle = {angle}')
        axis.set_aspect(1.0)

        #axis.plot(qp_v[0:1,phase_index,0], qp_v[0:1,phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1], color='blue')
        axis.plot(qp_P[phase_index,0], qp_P[phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot([0, qp_P[phase_index,0]], [0, qp_P[phase_index,1]], color='black', alpha=0.5)

        #axis.plot(segment_qp_v[0:1,phase_index,0], segment_qp_v[0:1,phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot(segment_qp_v[:,phase_index,0], segment_qp_v[:,phase_index,1], color='green')
        axis.plot(segment_qp_minus_P[phase_index,0], segment_qp_minus_P[phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot([0, segment_qp_minus_P[phase_index,0]], [0, segment_qp_minus_P[phase_index,1]], color='black', alpha=0.5)

        axis.plot(extrapolated_segment_qp_v[:,phase_index,0], extrapolated_segment_qp_v[:,phase_index,1], color='orange', alpha=0.5)

        axis = axis_vv[phase_index][1]
        axis.set_title(f'initial {s}z = {p_z_initial}\n(t, {s}z(t))\nblue:solution, green:extrapolated\nmax extrapolation error = {extrapolation_error}')

        axis.plot(t_v, qp_v[:,phase_index,2], color='blue')
        axis.plot(segment_t_v, segment_qp_v[:,phase_index,2], color='green')
        axis.plot(interpolated_segment_t_v, extrapolated_segment_qp_v[:,phase_index,2], color='orange', alpha=0.5)


    plot_p = pathlib.Path(f'SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_dilation.temp/qp.p_y={p_y_initial}.png')

    fig.tight_layout()
    plot_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_p), bbox_inches='tight')
    print(f'wrote to file "{plot_p}"')
    # VERY important to do this -- otherwise your memory will slowly fill up!
    # Not sure which one is actually sufficient -- apparently none of them are, YAY!
    plt.clf()
    plt.cla()
    plt.close()
    plt.close(fig)
    plt.close('all')
    del fig
    del axis_vv

def plot_dilating_extrapolated_trajectory (p_y_initial:float, lam:float) -> None:
    x_initial = 1.0
    y_initial = 0.0
    z_initial = 0.0
    p_x_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = heisenberg.self_similar.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    pickle_file_p = pathlib.Path(f'SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_dilation.temp/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = heisenberg.self_similar.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=15.0,
        solution_sheet=0,
    )

    #
    # Try to construct the t < 0 portion of the solution.
    #

    # Solve for the smallest P > 0 such that z(P) == 0 and P is a positively oriented zero of z..

    t_v = results.t_v
    qp_v = results.y_t
    z_v = qp_v[:,0,2]

    #print(z_v)

    z_zero_index_pair_v, z_zero_orientation_v, z_zero_t_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v, orientation_p=(lambda o:o < 0))
    #print(np.vstack((z_zero_index_pair_v.T, z_zero_orientation_v, z_zero_t_v)))
    assert len(z_zero_t_v) >= 2
    assert z_zero_t_v[0] == t_v[0]

    P_bound_index = z_zero_index_pair_v[1,1]
    P = z_zero_t_v[1]
    assert P <= t_v[P_bound_index]

    # TODO: Use Bezier interpolation instead
    qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
    qp_P = qp_interpolator(P)

    segment_t_v = -t_v[:P_bound_index+1]
    segment_qp_v = np.copy(qp_v[:P_bound_index+1,...])

    # Reverse these so that time goes forward.
    segment_t_v[...] = segment_t_v[::-1]
    segment_qp_v[...] = segment_qp_v[::-1,...]

    # Apply transformations to segment_qp_v (flip y, z, p_y, p_z, then flip p_x, p_y, p_z,
    # which is equivalent to flipping y, z, p_x).
    segment_qp_v[:,0,1] *= -1
    segment_qp_v[:,0,2] *= -1
    segment_qp_v[:,1,0] *= -1

    # TODO: Use Bezier interpolation instead
    segment_qp_interpolator = scipy.interpolate.interp1d(segment_t_v, segment_qp_v, axis=0)
    segment_qp_minus_P = segment_qp_interpolator(-P)

    angle_v = np.arctan2(qp_P[:,1], qp_P[:,0]) - np.arctan2(segment_qp_minus_P[:,1], segment_qp_minus_P[:,0])
    for i in range(len(angle_v)):
        while angle_v[i] < 0:
            angle_v[i] += 2*np.pi

    print(f'angle_v = {angle_v}')
    if np.max(angle_v) - np.min(angle_v) < 1.0e-6:
        print(f'angles matched as expected')
    else:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!\n! ANGLES DID NOT MATCH !\n!!!!!!!!!!!!!!!!!!!!!!!!')

    angle = np.mean(angle_v)

    R = np.array([
        [ np.cos(-angle), np.sin(-angle), 0.0],
        [-np.sin(-angle), np.cos(-angle), 0.0],
        [            0.0,            0.0, 1.0],
    ])
    extrapolated_segment_qp_v = np.einsum('ij,tpj->tpi', R, segment_qp_v)
    interpolated_segment_t_v = segment_t_v+2*P
    interpolated_segment_qp_v = qp_interpolator(interpolated_segment_t_v)

    extrapolation_error = np.max(np.abs(extrapolated_segment_qp_v - interpolated_segment_qp_v))

    print(f'extrapolation_error = {extrapolation_error}')
    if extrapolation_error > 5.0e-5:
        print('!!!!!!!!!!!!!!!!!!\n! ERROR EXCEEDED !\n!!!!!!!!!!!!!!!!!!')

    row_count   = 2
    col_count   = 2
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for phase_index in range(2):
        s = 'p_' if phase_index == 1 else ''
        angle = angle_v[phase_index]

        axis = axis_vv[phase_index][0]
        axis.set_title(f'initial ({s}x, {s}y) = {(p_x_initial, p_y_initial)}\n({s}x(t), {s}y(t))\nblue:solution, green:extrapolated\nangle = {angle}')
        axis.set_aspect(1.0)

        #axis.plot(qp_v[0:1,phase_index,0], qp_v[0:1,phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1], color='blue')
        axis.plot(qp_P[phase_index,0], qp_P[phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot([0, qp_P[phase_index,0]], [0, qp_P[phase_index,1]], color='black', alpha=0.5)

        #axis.plot(segment_qp_v[0:1,phase_index,0], segment_qp_v[0:1,phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot(segment_qp_v[:,phase_index,0], segment_qp_v[:,phase_index,1], color='green')
        axis.plot(segment_qp_minus_P[phase_index,0], segment_qp_minus_P[phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot([0, segment_qp_minus_P[phase_index,0]], [0, segment_qp_minus_P[phase_index,1]], color='black', alpha=0.5)

        axis.plot(extrapolated_segment_qp_v[:,phase_index,0], extrapolated_segment_qp_v[:,phase_index,1], color='orange', alpha=0.5)

        axis = axis_vv[phase_index][1]
        axis.set_title(f'initial {s}z = {p_z_initial}\n(t, {s}z(t))\nblue:solution, green:extrapolated\nmax extrapolation error = {extrapolation_error}')

        axis.plot(t_v, qp_v[:,phase_index,2], color='blue')
        axis.plot(segment_t_v, segment_qp_v[:,phase_index,2], color='green')
        axis.plot(interpolated_segment_t_v, extrapolated_segment_qp_v[:,phase_index,2], color='orange', alpha=0.5)


    plot_p = pathlib.Path(f'SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_dilation.temp/qp.p_y={p_y_initial}.png')

    fig.tight_layout()
    plot_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_p), bbox_inches='tight')
    print(f'wrote to file "{plot_p}"')
    # VERY important to do this -- otherwise your memory will slowly fill up!
    # Not sure which one is actually sufficient -- apparently none of them are, YAY!
    plt.clf()
    plt.cla()
    plt.close()
    plt.close(fig)
    plt.close('all')
    del fig
    del axis_vv

def compute_trajectory (p_x_initial:float, p_y_initial:float, base_dir_p:pathlib.Path) -> vorpy.integration.adaptive.IntegrateVectorFieldResults:
    pickle_file_p = base_dir_p / f'qp.p_x={p_x_initial}_p_y={p_y_initial}.pickle'
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    x_initial = 2.0
    y_initial = 0.0
    z_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = heisenberg.self_similar.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    return heisenberg.self_similar.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=100.0,
        solution_sheet=0,
    )

def plot_trajectories (results_v:typing.Sequence[vorpy.integration.adaptive.IntegrateVectorFieldResults], base_dir_p:pathlib.Path, *, result_index=None) -> None:
    row_count   = 2
    col_count   = 6
    size        = 8
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    def compute_collision_time (*, results:vorpy.integration.adaptive.IntegrateVectorFieldResults) -> float:
        # Don't bother if J is close to zero.
        J_initial = heisenberg.self_similar.kh.EuclideanNumerics.J__fast(results.y_t[0])
        if np.abs(J_initial) < 1.0e-6:
            return np.nan

        t_v = results.t_v
        z_v = results.y_t[:,0,2]
        _,_,z_zero_pos_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v, orientation_p=(lambda o:o>0))

        print(f'J_initial = {J_initial}, z_zero_pos_v = {z_zero_pos_v}')

        if len(z_zero_pos_v) < 3:
            print(f'exiting because len(z_zero_pos_v) (which was {len(z_zero_pos_v)}) was < 3')
            return np.nan

        z_zero_pos_delta_v = np.diff(z_zero_pos_v)
        lam_v = z_zero_pos_delta_v[1:] / z_zero_pos_delta_v[:-1]
        lam = np.mean(lam_v)

        # Too close to non-dilating to have a collision
        if np.abs(lam - 1.0) < 1.0e-3:
            print(f'exiting because lam (which was {lam}) was too close to 1')
            return np.nan

        collision_time = z_zero_pos_v[0] + z_zero_pos_delta_v[0] / (1.0 - lam)
        print(f'lam = {lam}, lam_v stddev: {np.std(lam_v)}, collision_time = {collision_time}')
        return collision_time

    for results in results_v:
        t_v = results.t_v
        qp_v = results.y_t

        p_x_initial = qp_v[0,1,0]
        p_y_initial = qp_v[0,1,1]

        collision_time = compute_collision_time(results=results)
        if np.isfinite(collision_time):
            # Assume that J is a "rate" of size / time.  Then with the fact that J is conserved
            # (assuming that H = 0), then the size must be changing linearly with time.
            J_initial = heisenberg.self_similar.kh.EuclideanNumerics.J__fast(results.y_t[0])

            precollision_mask_v = t_v < collision_time
            precollision_t_v = t_v[precollision_mask_v]
            if len(precollision_t_v) > 0:
                size_v = -J_initial * (collision_time - precollision_t_v)
                print(f'size_v[0] = {size_v[0]}')

                normalized_size_v = (collision_time - precollision_t_v) / (collision_time - precollision_t_v[0])

                axis = axis_vv[1][3]
                axis.set_title('"size"')
                axis.axhline(0.0, color='black')
                axis.plot(precollision_t_v, size_v)
                #axis.plot(precollision_t_v, normalized_size_v)
                axis.axvline(collision_time)
            else:
                axis = axis_vv[1][3]
                axis.set_title('"size" could not be computed')
                normalized_size_v = None

        def dilation (lam:float, qp:np.ndarray) -> np.ndarray:
            assert qp.shape == (2,3)
            retval = np.copy(qp)
            retval[0,0:2] *= lam
            retval[0,2]   *= lam**2
            retval[1,0:2] /= lam
            retval[1,2]   /= lam**2
            return retval

        if np.isfinite(collision_time) and normalized_size_v is not None:
            # Apply a progressive dilation (using sqrt(lam) because lam is based on z)
            precollision_qp_v = qp_v[precollision_mask_v,...]
            dilated_precollision_qp_v = np.copy(precollision_qp_v)
            for i,normalized_size in enumerate(normalized_size_v):
                dilated_precollision_qp_v[i] = dilation(1.0 / np.sqrt(normalized_size), dilated_precollision_qp_v[i])
        else:
            dilated_precollision_qp_v = None

        for phase_index in range(2):
            s = 'p_' if phase_index == 1 else ''

            axis = axis_vv[phase_index][0]
            axis.set_title(f'initial ({s}x, {s}y) = {(qp_v[0,phase_index,0], qp_v[0,phase_index,1])}\n({s}x(t), {s}y(t))')
            axis.set_aspect(1.0)
            if dilated_precollision_qp_v is not None:
                axis.plot(dilated_precollision_qp_v[:,phase_index,0], dilated_precollision_qp_v[:,phase_index,1])
            else:
                axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1])
            axis.plot([0.0], [0.0], '.', color='black')

            for coord_index,coord_name in enumerate(['x','y','z']):
                axis = axis_vv[phase_index][1+coord_index]
                axis.set_title(f'initial {s}{coord_name} = {qp_v[0,phase_index,coord_index]}\n(t, {s}{coord_name}(t))')
                if dilated_precollision_qp_v is not None:
                    axis.plot(t_v, dilated_precollision_qp_v[:,phase_index,coord_index])
                else:
                    axis.plot(t_v, qp_v[:,phase_index,coord_index])

        p_theta_v   = vorpy.apply_along_axes(heisenberg.self_similar.kh.EuclideanNumerics.p_theta__fast, (1,2), (results.y_t,))
        H_v         = vorpy.apply_along_axes(heisenberg.self_similar.kh.EuclideanNumerics.H__fast, (1,2), (results.y_t,))
        J_v         = vorpy.apply_along_axes(heisenberg.self_similar.kh.EuclideanNumerics.J__fast, (1,2), (results.y_t,))

        axis = axis_vv[0][4]
        axis.set_title(f'p_theta = {p_theta_v[0]}')
        axis.plot(t_v, p_theta_v)

        axis = axis_vv[1][4]
        axis.set_title(f'H = {H_v[0]}')
        axis.plot(t_v, H_v)

        axis = axis_vv[1][5]
        axis.set_title(f'J = {J_v[0]}')
        axis.plot(t_v, J_v)

    if len(results_v) == 1:
        plot_p = base_dir_p / f'qp.result_index={result_index}.x={results.y_t[0,0,0]}.y={results.y_t[0,0,1]}.z={results.y_t[0,0,2]}.p_x={results.y_t[0,1,0]}.p_y={results.y_t[0,1,1]}.p_z={results.y_t[0,1,2]}.png'
    else:
        plot_p = base_dir_p / f'results.png'

    fig.tight_layout()
    plot_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_p), bbox_inches='tight')
    print(f'wrote to file "{plot_p}"')
    # VERY important to do this -- otherwise your memory will slowly fill up!
    # Not sure which one is actually sufficient -- apparently none of them are, YAY!
    plt.clf()
    plt.cla()
    plt.close()
    plt.close(fig)
    plt.close('all')
    del fig
    del axis_vv

def plot_trajectories_QuadraticCylindrical (results_v:typing.Sequence[vorpy.integration.adaptive.IntegrateVectorFieldResults], plot_p:pathlib.Path) -> None:
    row_count   = 2
    col_count   = 4
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for results in results_v:
        t_v         = results.t_v
        qp_v        = results.y_t
        assert results.y_jet_to is not None
        qp_jet_t    = results.y_jet_to

        #qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
        qp_interpolator = vorpy.realfunction.bezier.cubic_interpolation(t_v, qp_jet_t)
        qp_initial  = qp_v[0]

        p_R_initial = qp_initial[1,0]
        H_initial   = heisenberg.self_similar.kh.QuadraticCylindricalNumerics.H__fast(qp_initial)
        J_initial   = heisenberg.self_similar.kh.QuadraticCylindricalNumerics.J__fast(qp_initial)

        theta_v     = qp_v[:,0,1]
        # Segment based on local maxima of theta_v
        _, _, theta_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(theta_v, t_v=t_v)
        print(f'theta_local_max_s_v = {theta_local_max_s_v}')
        #t_delta_v   = np.diff(theta_local_max_s_v)
        #scale_v     = t_delta_v[1:] / t_delta_v[:-1]

        # Find the sector bounds for the (t,R(t)) curve
        # Using local maxes and mins is very approximate.
        _,_,R_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(qp_v[:,0,0], t_v=t_v)
        _,_,R_local_min_s_v = vorpy.realfunction.piecewiselinear.local_minimizers(qp_v[:,0,0], t_v=t_v)
        print(f'R_local_max_s_v = {R_local_max_s_v}')
        print(f'R_local_min_s_v = {R_local_min_s_v}')
        R_local_max_v = qp_interpolator(R_local_max_s_v)[:,0,0]
        R_local_min_v = qp_interpolator(R_local_min_s_v)[:,0,0]
        print(f'R_local_max_v = {R_local_max_v}')
        print(f'R_local_min_v = {R_local_min_v}')
        R_sector_upper_bound_slope = np.mean(np.diff(R_local_max_v) / np.diff(R_local_max_s_v))
        R_sector_lower_bound_slope = np.mean(np.diff(R_local_min_v) / np.diff(R_local_min_s_v))
        R_sector_middle_slope = np.mean([R_sector_upper_bound_slope, R_sector_lower_bound_slope])

        # Find the sector bounds for the (t,w(t)) curve
        # Using local maxes and mins is very approximate.
        _,_,w_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(qp_v[:,0,2], t_v=t_v)
        _,_,w_local_min_s_v = vorpy.realfunction.piecewiselinear.local_minimizers(qp_v[:,0,2], t_v=t_v)
        print(f'w_local_max_s_v = {w_local_max_s_v}')
        print(f'w_local_min_s_v = {w_local_min_s_v}')
        w_local_max_v = qp_interpolator(w_local_max_s_v)[:,0,2]
        w_local_min_v = qp_interpolator(w_local_min_s_v)[:,0,2]
        print(f'w_local_max_v = {w_local_max_v}')
        print(f'w_local_min_v = {w_local_min_v}')
        w_sector_upper_bound_slope = np.mean(np.diff(w_local_max_v) / np.diff(w_local_max_s_v))
        w_sector_lower_bound_slope = np.mean(np.diff(w_local_min_v) / np.diff(w_local_min_s_v))
        w_sector_middle_slope = np.mean([w_sector_upper_bound_slope, w_sector_lower_bound_slope])

        print(f'QuadraticCylindrical -- qp_initial:\n{qp_initial}')
        print(f'H_initial = {H_initial}')
        print(f'J_initial = {J_initial}')
        #print(f'scale_v   = {scale_v}')

        #scale       = np.mean(scale_v)
        #scale_error = np.max(scale_v) - np.min(scale_v)

        #print(f'scale_error = {scale_error}')
        #print(f'scale       = {scale}')

        # Take each segment and un-dilate it using an exponentially decreasing scale factor
        R_segment_v     = qp_interpolator(theta_local_max_s_v)[:,0,0]
        r_segment_v     = np.sqrt(R_segment_v)

        if len(theta_local_max_s_v) >= 2:
            # The correspondence between s (time parameter of dilating orbit) and t (time parameter
            # of non-dilating orbit) is
            #
            #     t = P*log_{lam}((lam-1)*s/(P*lam)),
            #
            # where lam is the scale factor and P is the period in the t time parameter, and log_{lam}
            # denotes the base-lam logarithm.

            # Pick arbitrary period for now, to be solved for later.
            #P = 1.0
            #P = theta_local_max_s_v[1] - theta_local_max_s_v[0] # This is semi-arbitrary
            theta_local_max_s_diff_v = np.diff(theta_local_max_s_v)
            lam_v = theta_local_max_s_diff_v[1:] / theta_local_max_s_diff_v[:-1]
            lam = np.mean(lam_v)
            lam_spread = np.max(lam_v) - np.min(lam_v)

            print(f'lam_v = {lam_v}')
            print(f'lam = {lam}')
            print(f'lam_spread = {lam_spread}')

            if lam < 1.0:
                # Collision is in the future
                s_collision = theta_local_max_s_v[0] + (theta_local_max_s_v[1] - theta_local_max_s_v[0])/(1.0-lam)
            elif lam > 1.0:
                # Collision is in the past -- TODO LATER
                s_collision = 0.0
            else: # lam == 1.0
                # No collision
                s_collision = 0.0

            # TODO: Actually solve for the correct function
            P = (theta_local_max_s_v[1] - theta_local_max_s_v[0]) / lam

            def tau (s:float) -> float:
                return P*np.log((lam-1.0)*(s-s_collision)/(P*lam))/np.log(lam)

            tau_theta_local_max_s_v = np.vectorize(tau)(theta_local_max_s_v)
            diff_tau_theta_local_max_s_v = np.diff(tau_theta_local_max_s_v)
            tau_diff_theta_local_max_spread = np.max(diff_tau_theta_local_max_s_v) - np.min(diff_tau_theta_local_max_s_v)
            print(f'tau_theta_local_max_s_v = {diff_tau_theta_local_max_s_v}')
            print(f'tau_diff_theta_local_max_spread = {tau_diff_theta_local_max_spread}')

        else:
            lam_v       = []
            lam         = np.nan
            lam_spread  = np.nan

            def tau (s:float) -> float:
                return s

        print(f'lam_v = {lam_v}')
        print(f'lam = {lam}')
        print(f'lam_spread = {lam_spread}')

        w_sector_middle_line = np.vectorize(lambda s:(s - s_collision) * w_sector_middle_slope)

        if len(r_segment_v) >= 2:
            r_segment_jet_t         = np.ndarray((len(r_segment_v),2), dtype=float)
            r_segment_jet_t[:,0]    = r_segment_v
            r_segment_jet_t[0,1]    = (r_segment_v[1] - r_segment_v[0]) / (theta_local_max_s_v[1] - theta_local_max_s_v[0])
            r_segment_jet_t[-1,1]   = (r_segment_v[-1] - r_segment_v[-2]) / (theta_local_max_s_v[-1] - theta_local_max_s_v[-2])
            r_segment_jet_t[1:-1,1] = (r_segment_v[2:] - r_segment_v[:-2]) / (theta_local_max_s_v[2:] - theta_local_max_s_v[:-2])
            scale_interpolator      = vorpy.realfunction.bezier.cubic_interpolation(theta_local_max_s_v, r_segment_jet_t)
            print('USING CUBIC INTERPOLATION FOR SCALE')
        else:
            scale_interpolator      = scipy.interpolate.interp1d(theta_local_max_s_v, r_segment_v)
            print('USING LINEAR INTERPOLATION FOR SCALE')

        unwrapped_mask_v    = (t_v >= theta_local_max_s_v[0]) & (t_v <= theta_local_max_s_v[-1])
        unwrapped_s_v       = t_v[unwrapped_mask_v]
        unwrapped_t_v       = np.vectorize(tau)(unwrapped_s_v) + theta_local_max_s_v[0]
        unwrapped_qp_v      = np.copy(qp_v[unwrapped_mask_v])
        scale_v             = scale_interpolator(unwrapped_s_v)
        unwrapped_qp_v[:,0,0] /= scale_v**2
        unwrapped_qp_v[:,0,2] -= w_sector_middle_line(unwrapped_s_v)
        unwrapped_qp_v[:,0,2] /= scale_v**2
        unwrapped_qp_v[:,1,0] *= scale_v**2
        unwrapped_qp_v[:,1,2] *= scale_v**2

        H_unwrapped_qp_v = vorpy.apply_along_axes(heisenberg.self_similar.kh.QuadraticCylindricalNumerics.H__fast, (1,2), (unwrapped_qp_v,))
        J_unwrapped_qp_v = vorpy.apply_along_axes(heisenberg.self_similar.kh.QuadraticCylindricalNumerics.J__fast, (1,2), (unwrapped_qp_v,))

        print(f'min, max of H_unwrapped_qp_v: {np.min(H_unwrapped_qp_v), np.max(H_unwrapped_qp_v)}')
        print(f'min, max of J_unwrapped_qp_v: {np.min(J_unwrapped_qp_v), np.max(J_unwrapped_qp_v)}')

        euclidean_qp_v = vorpy.apply_along_axes(heisenberg.self_similar.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (qp_v,))

        euclidean_unwrapped_qp_v = vorpy.apply_along_axes(heisenberg.self_similar.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (unwrapped_qp_v,))

        # Compute a trajectory using the unwrapped initial condition
        transformed_results = heisenberg.self_similar.kh.QuadraticCylindricalNumerics.compute_trajectory(
            pathlib.Path(str(plot_p) + '.transformed.pickle'),
            unwrapped_qp_v[0],
            t_final=100.0,
            solution_sheet=0,
            return_y_jet=False,
        )

        euclidean_transformed_qp_v = vorpy.apply_along_axes(heisenberg.self_similar.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (transformed_results.y_t,))

        for phase_index in range(2):
            s = 'p_' if phase_index == 1 else ''

            axis = axis_vv[phase_index][0]
            axis.set_title(f'initial ({s}x, {s}y) = {(euclidean_qp_v[0,phase_index,0], euclidean_qp_v[0,phase_index,1])}\n({s}x(t), {s}y(t))')
            axis.set_aspect(1.0)
            axis.plot(euclidean_qp_v[:,phase_index,0], euclidean_qp_v[:,phase_index,1])
            if p_R_initial != 0.0:
                axis.plot(euclidean_unwrapped_qp_v[:,phase_index,0], euclidean_unwrapped_qp_v[:,phase_index,1])
                axis.plot(euclidean_transformed_qp_v[:,phase_index,0], euclidean_transformed_qp_v[:,phase_index,1], color='green')

            axis = axis_vv[phase_index][1]
            axis.set_title(f'initial {s}R = {qp_v[0,phase_index,0]}\n(t, {s}R(t))')
            axis.plot(t_v, qp_v[:,phase_index,0])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,0])
                axis.plot(transformed_results.t_v, transformed_results.y_t[:,phase_index,0], color='green')
            if phase_index == 0:
                axis.axhline(0, color='black')
                # Draw the lines defining the sector that bounds the curve (t,R(t))
                s_vals = np.array(axis.get_xlim())
                R_vals = (s_vals - s_collision) * R_sector_lower_bound_slope
                axis.plot(s_vals, R_vals, color='red')
                R_vals = (s_vals - s_collision) * R_sector_middle_slope
                axis.plot(s_vals, R_vals, color='green')
                R_vals = (s_vals - s_collision) * R_sector_upper_bound_slope
                axis.plot(s_vals, R_vals, color='blue')

            axis = axis_vv[phase_index][2]
            axis.set_title(f'initial {s}theta = {qp_v[0,phase_index,1]}\n(t, {s}theta(t))')
            axis.plot(t_v, qp_v[:,phase_index,1])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,1])
                axis.plot(transformed_results.t_v, transformed_results.y_t[:,phase_index,1], color='green')

            axis = axis_vv[phase_index][3]
            axis.set_title(f'initial {s}w = {qp_v[0,phase_index,2]}\n(t, {s}w(t))')
            axis.plot(t_v, qp_v[:,phase_index,2])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,2])
                axis.plot(transformed_results.t_v, transformed_results.y_t[:,phase_index,2], color='green')
            if phase_index == 0:
                axis.axhline(0, color='black')
                # Draw the lines defining the sector that bounds the curve (t,w(t))
                s_vals = np.array(axis.get_xlim())
                R_vals = (s_vals - s_collision) * w_sector_lower_bound_slope
                axis.plot(s_vals, R_vals, color='red')
                R_vals = (s_vals - s_collision) * w_sector_middle_slope
                axis.plot(s_vals, R_vals, color='green')
                R_vals = (s_vals - s_collision) * w_sector_upper_bound_slope
                axis.plot(s_vals, R_vals, color='blue')

    #plot_p = base_dir_p / f'qp.p_y={p_y_initial}.png'
    plot_p.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    plt.savefig(str(plot_p), bbox_inches='tight')
    print(f'wrote to file "{plot_p}"')
    # VERY important to do this -- otherwise your memory will slowly fill up!
    # Not sure which one is actually sufficient -- apparently none of them are, YAY!
    plt.clf()
    plt.cla()
    plt.close()
    plt.close(fig)
    plt.close('all')
    del fig
    del axis_vv

def transform_J (p_y_initial:float, other_trajectory_p_x_initial:float) -> None:
    base_dir_p = pathlib.Path(f'SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_dilation.transform_J')

    # TODO: Find a J = 0 orbit and a J != 0 orbit having the same delta in theta for each quasiperiod.
    # Is this the same as identical angular momentum (i.e. p_theta)?

    zero_J_trajectory_results = compute_trajectory(0.0, p_y_initial, base_dir_p)



    nonzero_J_trajectory_results = compute_trajectory(other_trajectory_p_x_initial, p_y_initial, base_dir_p)

    plot_trajectories([zero_J_trajectory_results, nonzero_J_trajectory_results], base_dir_p)

def unwrap_dilating_trajectory (p_R_initial:float, p_theta_initial:float, base_dir_p:pathlib.Path) -> None:
    pickle_file_p = base_dir_p / f'qp.p_R={p_R_initial}_p_theta={p_theta_initial}.pickle'
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    R_initial = 1.0
    theta_initial = 0.0
    w_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = heisenberg.self_similar.kh.QuadraticCylindricalNumerics.qp_constrained_by_H__fast(
        np.array([R_initial, theta_initial, w_initial, p_R_initial, p_theta_initial, H_initial])
    )[solution_sheet]

    results = heisenberg.self_similar.kh.QuadraticCylindricalNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=100.0,
        solution_sheet=0,
        return_y_jet=True,
    )

    plot_p = base_dir_p / f'qp.p_R={p_R_initial}_p_theta={p_theta_initial}.png'
    plot_trajectories_QuadraticCylindrical([results], plot_p)

#def plot_related_trajectories (*, p_theta_initial:float, J_initial_v:typing.List[float], plot_dir_p:pathlib.Path):
def plot_related_trajectories (*, x_initial:float, p_x_initial_v:typing.List[float], p_y_initial:float, plot_dir_p:pathlib.Path):
    solution_sheet = 0

    #def compute_qp_initial (*, p_theta_initial:float, J_initial:float) -> np.ndarray:
        ## Use Euclidean coordinates
        #x_initial = 1.0
        #y_initial = 0.0
        #z_initial = 0.0
        #p_x_initial = J_initial
        #p_y_initial = p_theta_initial
        ## TODO: Plot both sheets
        #if solution_sheet == 0:
            #p_z_initial = -2.0*p_theta_initial + np.sqrt(1.0/np.pi - 4.0*J_initial**2)
        #else:
            #p_z_initial = -2.0*p_theta_initial - np.sqrt(1.0/np.pi - 4.0*J_initial**2)

        #qp_initial = np.array([
            #[x_initial, y_initial, z_initial],
            #[p_x_initial, p_y_initial, p_z_initial],
        #])

        #H_initial = heisenberg.self_similar.kh.EuclideanNumerics.H__fast(qp_initial)
        #assert np.abs(H_initial) < 1.0e-14

        #return qp_initial

    plot_dir_p.mkdir(parents=True, exist_ok=True)
    t_final = 200.0
    results_v = []
    #for J_initial in J_initial_v:
    for p_x_initial in p_x_initial_v:
        #qp_initial = compute_qp_initial(p_theta_initial=p_theta_initial, J_initial=J_initial)
        qp_initial = heisenberg.self_similar.kh.EuclideanNumerics.qp_constrained_by_H__fast(np.array([x_initial, 0.0, 0.0, p_x_initial, p_y_initial, 0.0]))[0]

        results = heisenberg.self_similar.kh.EuclideanNumerics.compute_trajectory(
            #plot_dir_p / f'p_theta={p_theta_initial}.J={J_initial}.pickle',
            plot_dir_p / f'x={x_initial}.p_x={p_x_initial}.p_y={p_y_initial}.pickle',
            qp_initial,
            t_final=t_final,
            solution_sheet=solution_sheet, # This isn't used in the computation, it's just stored in the pickle file
            return_y_jet=False,
        )
        results_v.append(results)

    plot_trajectories(results_v=results_v, base_dir_p=plot_dir_p)

def plot_z_axis_trajectories (*, z_initial_v:typing.List[float], p_z_initial_v:typing.List[float], plot_dir_p:pathlib.Path):
    solution_sheet = 0

    x_initial = 0.0
    y_initial = 0.0
    p_y_initial = 0.0

    plot_dir_p.mkdir(parents=True, exist_ok=True)
    t_final = 2000.0
    results_v = []
    for z_initial in z_initial_v:
        p_x_initial = 0.25 / np.sqrt(np.pi*z_initial)
        for p_z_initial in p_z_initial_v:
            qp_initial = np.array([
                [x_initial,   y_initial,   z_initial],
                [p_x_initial, p_y_initial, p_z_initial]
            ])

            results = heisenberg.self_similar.kh.EuclideanNumerics.compute_trajectory(
                #plot_dir_p / f'p_theta={p_theta_initial}.J={J_initial}.pickle',
                plot_dir_p / f'z={z_initial}.p_z={p_z_initial}.pickle',
                qp_initial,
                t_final=t_final,
                solution_sheet=solution_sheet, # This isn't used in the computation, it's just stored in the pickle file
                return_y_jet=False,
            )
            plot_trajectories(results_v=[results], base_dir_p=plot_dir_p, result_index=len(results_v))
            results_v.append(results)

    plot_trajectories(results_v=results_v, base_dir_p=plot_dir_p)

def main ():
    import sys

    #for p_y_initial in np.linspace(0.05, 0.4, 20):
        #plot_J_equal_zero_extrapolated_trajectory(p_y_initial)



    #p_y_initial = 0.1
    #plot_J_equal_zero_extrapolated_trajectory(p_y_initial)



    #other_trajectory_p_x_initial = -0.1
    #for p_y_initial in np.linspace(0.05, 0.4, 20):
        #transform_J(p_y_initial, other_trajectory_p_x_initial)



    #base_dir_p = pathlib.Path('unwrap.06')
    ##for p_R_initial,p_theta_initial in itertools.product(np.linspace(-1.0/64, 1.0/64, 3), np.linspace(0.05, 0.4, 2)):
    #for p_R_initial,p_theta_initial in itertools.product(np.linspace(-1.0/64, 0.0, 2, endpoint=False), np.linspace(0.05, 0.4, 2)):
        #try:
            #unwrap_dilating_trajectory(p_R_initial, p_theta_initial, base_dir_p)
        #except ValueError as e:
            #print(f'Caught {e}')
            #pass



    #J_bound     = 0.0625
    ##J_bound     = 0.25
    #J_count     = 9
    ##J_initial_v = np.linspace(-J_bound, 0.0, J_count//2, endpoint=False).tolist() + np.linspace(0.0, J_bound, (J_count+1)//2).tolist()
    #J_initial_v = np.linspace(-J_bound, 0.0, J_count).tolist()
    #plot_related_trajectories(
        #p_theta_initial=0.2,
        #J_initial_v=J_initial_v,
        #plot_dir_p=pathlib.Path('related.01'),
    #)



    #print(f'X_H:\n{heisenberg.self_similar.kh.QuadraticCylindricalSymbolics.X_H__symbolic(heisenberg.self_similar.kh.QuadraticCylindricalSymbolics.qp_coordinates()).reshape(6,1)}')
    #sys.exit(0)

    #p_x_bound     = 0.0625
    ##p_x_bound     = 0.25
    #p_x_count     = 9
    #p_x_initial_v = np.linspace(-p_x_bound, 0.0, p_x_count).tolist()
    #plot_related_trajectories(
        #x_initial=2.0,
        #p_x_initial_v=p_x_initial_v,
        #p_y_initial=0.2,
        #plot_dir_p=pathlib.Path('related.01'),
    #)

    z_initial_v = [1.0]
    #p_z_initial_v = [0.0]+[0.5**p for p in np.linspace(0.0, 5.0, num=33)]
    #p_z_initial_v = np.linspace(0.127, 0.158, num=16)
    p_z_initial_v = np.linspace(0.0, 1.0, num=33)
    plot_z_axis_trajectories(z_initial_v=z_initial_v, p_z_initial_v=p_z_initial_v, plot_dir_p=pathlib.Path('SelfSimilarityInTheKeplerHeisenbergProblem/generated-data/kh_z_axis_png_plots'))

if __name__ == '__main__':
    main()
