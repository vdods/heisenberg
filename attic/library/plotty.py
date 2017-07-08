# Single-function frontend to pylab's plot
def plotty (*args, **kwargs):
    assert 'save_to_file' in kwargs, "Must specify a filename in keyword argument save_to_file to save the plot to."
    import pylab
    pylab.plot(*args)
    if 'title' in kwargs:
        pylab.title(kwargs['title'])
    if 'grid' in kwargs:
        pylab.grid(kwargs['grid'])
    pylab.savefig(kwargs['save_to_file'])
    pylab.show()

def plotty_2d_points (points, **kwargs):
    """
    Call plotty(Xs, Ys, **kwargs), where Xs and Ys are the lists derived from the
    x_index'th and y_index'th components of each element of points respectively.

    The keyword arguments are:

    - x_index (optional) : The index of the point component to use for the X-axis values.
                           The default value is 0.
    - y_index (optional) : The index of the point component to use for the Y-axis values.
                           The default value is 1.
    """
    x_index = kwargs.get('x_index', 0)
    y_index = kwargs.get('y_index', 1)
    Xs = [point[x_index] for point in points]
    Ys = [point[y_index] for point in points]
    plotty(Xs, Ys, **kwargs)
