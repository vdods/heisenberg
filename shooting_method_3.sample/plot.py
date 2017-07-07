import glob
import numpy as np
import sys
import vorpy.pickle

def get_stuff (range_v):
    pickle_filename_v               = glob.glob('./sample_v.*.pickle')
    point_v                         = []
    for pickle_filename in pickle_filename_v:
        sample_v                    = vorpy.pickle.unpickle(pickle_filename=pickle_filename, log_out=sys.stdout)
        for sample in sample_v:
            initial2                = sample[0]
            objective               = sample[4]
            t_min                   = sample[5]
            max_abs_H               = sample[6]
            max_abs_J_minus_J_0     = sample[7]
            if range_v[0] <= objective < range_v[1]:
                point_v.append(np.array((initial2[0], initial2[1], objective, t_min, max_abs_H, max_abs_J_minus_J_0)))
    return np.array(point_v)

#range_vv = [
    #(-np.inf, 0.0),
    #(0.0, 1.0e-3),
    #(1.0e-3, 1.0e-2),
    #(1.0e-2, 1.0e-1),
    #(1.0e-1, 1.0e10),
#]
#point_vv = [get_stuff(range_v) for range_v in range_vv]
#color_vv = [
    #(255,   0,   0, 255),
    #(180, 180,   0, 255),
    #(  0, 255,   0, 255),
    #(  0, 180, 180, 255),
    #(  0,   0, 255, 255),
#]
point_v = get_stuff((1.0e-16, np.inf))
print('number of points: {0}'.format(point_v.shape[0]))

#if __name__ == '__main__':
    #print(get_stuff(0.1))
    #sys.exit(0)

#if __name__ == '__main__':
    #do_stuff()



from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(1200,1200)
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('(p_x,p_y) initial condition scatterplot')

## create areas to add plots
w1 = view.addPlot()
w2 = view.addPlot()
view.nextRow()
w3 = view.addPlot()
w4 = view.addPlot()

## Make all plots clickable
lastClicked = []
def clicked(plot, points):
    global lastClicked
    for p in lastClicked:
        p.resetPen()
    print("clicked points", points)
    for p in points:
        p.setPen('b', width=2)
    lastClicked = points

def color_scatterplot (plot, point_v, value_v, *, use_log=False):
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

color_scatterplot(w1, point_v[:,0:2], point_v[:,2], use_log=True) # objective
color_scatterplot(w2, point_v[:,0:2], point_v[:,3], use_log=False) # t_min
color_scatterplot(w3, point_v[:,0:2], point_v[:,4], use_log=True) # max_abs_H
color_scatterplot(w4, point_v[:,0:2], point_v[:,5], use_log=True) # max_abs_J_minus_J_0

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

