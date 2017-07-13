"""
TODO:
-   Feature: Clicking on a point in the parameter space plots the integral curve with that initial condition
    so that the parameter space can be explored interactively.
-   Feature: Link the x axes for all the plots in 1D embedding domain.
-
"""

import glob
import heisenberg.library.util
import numpy as np
import os
import pyqtgraph as pg
import pyqtgraph.Qt
import sys
import vorpy.pickle

subprogram_description = 'Provides visualization of the data generated by the heisenberg.sample subprogram.  In particular, this gives a colormapped scatterplot of the objective function on the fully reduced, 2-parameter initial condition space.'

def read_sample_pickles (samples_dir, range_v):
    glob_pattern                    = os.path.join(samples_dir, 'sample_v.*.pickle')
    print('glob_pattern = "{0}"'.format(glob_pattern))
    pickle_filename_v               = glob.glob(glob_pattern)
    print(pickle_filename_v)
    data_v                          = []
    dimension_d                     = {1:0, 2:0}
    for pickle_filename in pickle_filename_v:
        pickle_data                 = vorpy.pickle.unpickle(pickle_filename=pickle_filename, log_out=sys.stdout)
        # TEMP legacy compatibility
        if type(pickle_data) == list:
            sample_v                = pickle_data
        elif type(pickle_data) == dict:
            sample_v                = pickle_data['sample_v']
        else:
            assert False, 'unknown data type {0} found in pickle'.format(type(pickle_data))
        for sample in sample_v:
            initial                 = sample[0]
            objective               = sample[4]
            t_min                   = sample[5]
            max_abs_H               = sample[6]
            max_abs_J_minus_J_0     = sample[7]
            if range_v[0] <= objective < range_v[1]:
                # TEMP HACK -- probably just use a different function altogether, or use a different data structure
                if initial.shape == (1,):
                    dimension_d[1] += 1
                    data_v.append(np.array((objective, t_min, max_abs_H, max_abs_J_minus_J_0, initial[0])))
                else:
                    dimension_d[2] += 1
                    data_v.append(np.array((objective, t_min, max_abs_H, max_abs_J_minus_J_0, initial[0], initial[1])))

    assert dimension_d[1] == 0 or dimension_d[2] == 0, 'inhomogeneous data (mixed dimensions)'
    dimension = 1 if dimension_d[1] > 0 else 2

    if len(data_v) == 0:
        print('No data found in "{0}" files.'.format(glob_pattern))
        return None, dimension
    else:
        return np.array(data_v), dimension

def plot_samples (dynamics_context, options, *, rng):
    data_v,dimension = read_sample_pickles(options.samples_dir, (1.0e-16, np.inf))
    if data_v is None:
        return

    print('number of points: {0}'.format(data_v.shape[0]))

    app = pyqtgraph.Qt.QtGui.QApplication([])
    mw = pyqtgraph.Qt.QtGui.QMainWindow()
    mw.resize(1200,1200)
    view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle('(p_x,p_y) initial condition scatterplot')

    ## create areas to add plots
    w1 = view.addPlot(name='w1', title='objective')
    w2 = view.addPlot(name='w2', title='max(abs(H))')
    view.nextRow()
    w3 = view.addPlot(name='w3', title='t_min')
    w4 = view.addPlot(name='w4', title='max(abs(J-J(0)))')

    ## Make all plots clickable
    lastClicked = []
    def clicked(plot, points):
        global lastClicked
        for p in lastClicked:
            p.resetPen()
        print('clicked points', points)
        for p in points:
            p.setPen('b', width=2)
        lastClicked = points

    if dimension == 1:
        # Compute all local minima of the objective function
        p_y_v                   = data_v[:,4]
        objective_v             = data_v[:,0]
        local_min_index_v       = [i for i in range(1,len(objective_v)-1) if objective_v[i-1] > objective_v[i] and objective_v[i] < objective_v[i+1]]
        # Use quadratic fit to compute time of local mins at sub-sample accuracy.
        local_min_v             = []
        for local_min_index in local_min_index_v:
            s                   = slice(local_min_index-1, local_min_index+2)
            p_y_min,objective   = heisenberg.library.util.quadratic_min_time_parameterized(p_y_v[s], objective_v[s])
            local_min_v.append((p_y_min,objective))
        print('local mins of objective function in (p_y, objective) form:')
        for local_min in local_min_v:
            print('    {0}'.format(local_min))

        def scatterplot (plot, point_v, value_v, *, use_log=False):
            assert np.all(np.isfinite(point_v))
            filter_v = np.isfinite(value_v)

            filtered_point_v = point_v[filter_v]
            filtered_value_v = value_v[filter_v]

            brush = pg.mkBrush(255, 255, 255, 255)
            s = pg.ScatterPlotItem(size=2, brush=brush)
            plot.addItem(s)
            s.addPoints(x=filtered_point_v, y=filtered_value_v)
            s.sigClicked.connect(clicked)
            plot.setLogMode(x=False, y=use_log)
            return s

        def lineplot (plot, point_v, value_v, *, use_log=False):
            assert np.all(np.isfinite(point_v))
            filter_v = np.isfinite(value_v)

            filtered_point_v = point_v[filter_v]
            filtered_value_v = value_v[filter_v]

            plot.plot(filtered_point_v, filtered_value_v)
            plot.setLogMode(x=False, y=use_log)

        ##scatterplot(w1, data_v[:,4], data_v[:,0], use_log=True) # objective
        #scatterplot(w1, data_v[:,4], data_v[:,0], use_log=False) # objective
        #scatterplot(w2, data_v[:,4], data_v[:,2], use_log=True) # max_abs_H
        scatterplot(w3, data_v[:,4], data_v[:,1], use_log=False) # t_min
        #scatterplot(w4, data_v[:,4], data_v[:,3], use_log=True) # max_abs_J_minus_J_0

        lineplot(w1, data_v[:,4], data_v[:,0], use_log=False) # objective
        lineplot(w2, data_v[:,4], data_v[:,2], use_log=False) # max_abs_H
        #lineplot(w3, data_v[:,4], data_v[:,1], use_log=False) # t_min
        lineplot(w4, data_v[:,4], data_v[:,3], use_log=False) # max_abs_J_minus_J_0

        # Link all plots' x axes together
        w2.setXLink('w1')
        w3.setXLink('w1')
        w4.setXLink('w1')
    elif dimension == 2:
        def color_scatterplot_2d (plot, point_v, value_v, *, use_log=False):
            if use_log:
                func = np.log
            else:
                func = lambda x:x

            assert np.all(np.isfinite(point_v))
            filter_v = np.isfinite(value_v)

            filtered_point_v = point_v[filter_v]
            filtered_value_v = value_v[filter_v]

            low = np.nanmin(func(filtered_value_v))
            high = np.nanmax(func(filtered_value_v))
            divisor = high - low
            print('low = {0}, high = {1}, divisor = {2}'.format(low, high, divisor))

            def brush_from_objective (objective):
                parameter = (func(objective) - low) / divisor
                return pg.mkBrush(int(round(255*parameter)), int(round(255*(1.0-parameter))), 0, 255)

            s = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))#, brush=pg.mkBrush(255, 255, 255, 128))
            plot.addItem(s)
            s.addPoints(x=filtered_point_v[:,0], y=filtered_point_v[:,1], brush=[brush_from_objective(objective) for objective in filtered_value_v])
            s.sigClicked.connect(clicked)
            return s

        color_scatterplot_2d(w1, data_v[:,4:6], data_v[:,0], use_log=True) # objective
        color_scatterplot_2d(w2, data_v[:,4:6], data_v[:,1], use_log=False) # t_min
        color_scatterplot_2d(w3, data_v[:,4:6], data_v[:,2], use_log=True) # max_abs_H
        color_scatterplot_2d(w4, data_v[:,4:6], data_v[:,3], use_log=True) # max_abs_J_minus_J_0
    else:
        assert False, 'dimension = {0}, which should never happen'.format(dimension)

    ## Start Qt event loop unless running in interactive mode.
    if (sys.flags.interactive != 1) or not hasattr(pyqtgraph.Qt.QtCore, 'PYQT_VERSION'):
        pyqtgraph.Qt.QtGui.QApplication.instance().exec_()

