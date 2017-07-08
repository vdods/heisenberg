import itertools
import numpy as np

def geometric (x, degree):
    """Returns the geometric series (1, x, x**2, ..., x**degree) as a numpy.ndarray."""
    return np.fromiter((x**d for d in xrange(degree+1)), type(x), count=degree+1)

def python_function_from_coefficients (coefficients):
    def python_function (x):
        return np.dot(coefficients, geometric(x,len(coefficients)-1))
    return python_function

def fit (samples, degree, sample_weights=None):
    """
    Fit a univariate polynomial function to the 2d points given in samples, where the rows of
    samples are the points.  The return value is the vector of coefficients of the polynomial
    (see p below) which minimizes the squared error of the polynomial at the given samples.

    Denote the components of samples as

        samples = [
            [x[0], y[0]],
            [x[1], y[1]],
            ...
        ]

    and let

        p(coefficients)(t) = sum(coefficient*t**i for i,coefficient in enumerate(coefficients))

    noting that coefficients[0] is the constant term, coefficients[1] is the linear coefficient, etc.
    """
    assert len(samples.shape) == 2
    assert samples.shape[1] == 2, 'Expected the rows of samples to be (x,y) pairs.'

    A = np.zeros((degree+1,degree+1), dtype=float)
    B = np.zeros((degree+1,), dtype=float)
    weight_iterator = sample_weights if sample_weights is not None else itertools.cycle([1.0])
    for (x,y),weight in itertools.izip(samples,weight_iterator):
        g = geometric(x, degree)
        A += weight*np.outer(g,g)
        B += weight*y*g
    coefficients,_,_,_ = np.linalg.lstsq(A,B)
    return coefficients

if __name__ == '__main__':
    import itertools
    import matplotlib.pyplot as plt
    import scipy.signal
    import sys
    import warnings

    warnings.filterwarnings('ignore', module='matplotlib')

    def unzip (*zipped):
        return zip(*zipped)

    def test0 ():
        sys.stdout.write('polynomial.test0()\n')

        sys.stdout.write('    linear ... ')
        samples = np.array([
            [0.0, 8.0],
            [2.0, 9.0],
        ])
        coefficients = fit(samples, 1)
        assert np.all(np.abs(coefficients - np.array([8.5, 0.5])))
        sys.stdout.write('passed.\n')

        sys.stdout.write('    quadratic ... ')
        samples = np.array([
            [-1.0, 1.0],
            [ 0.0, 0.0],
            [ 1.0, 1.0]
        ])
        coefficients = fit(samples, 2)
        assert np.all(np.abs(coefficients - np.array([0.0, 0.0, 1.0])) < 1.0e-12)
        sys.stdout.write('passed.\n')

    def test_fit (test_name, domain, signal, sample_weights=None):
        samples = np.vstack((domain, signal)).T

        highest_degree = 5
        fit_degree_v = np.linspace(0, highest_degree, highest_degree+1, dtype=int)
        fit_coefficients_v = [fit(samples, degree, sample_weights=sample_weights) for degree in fit_degree_v]
        assert len(fit_degree_v) == len(fit_coefficients_v)
        geometric_of_domain = np.array([
            geometric(x, highest_degree) for x in domain
        ])
        fit_signal_v = np.array([
            np.einsum('i,ji', fit_coefficients_v[i], geometric_of_domain[:,:fit_degree+1])
            for i,fit_degree in enumerate(fit_degree_v)
        ])

        row_count = len(fit_degree_v)+1 if sample_weights is not None else len(fit_degree_v)
        col_count = 1
        fig,axes = plt.subplots(row_count, col_count, squeeze=False, figsize=(10*col_count,5*row_count))

        for fit_index,(fit_degree,fit_signal) in enumerate(itertools.izip(fit_degree_v,fit_signal_v)):
            axis = axes[fit_index][0]
            axis.set_title('polynomial fit of degree {0}'.format(fit_degree))
            axis.plot(domain, signal, lw=7, color='green', alpha=0.2)
            axis.plot(domain, fit_signal, color='black')

        if sample_weights is not None:
            axis = axes[-1][0]
            axis.set_title('sample_weights')
            axis.plot(domain, sample_weights)

        fig.tight_layout()
        filename = 'polynomial.{0}.png'.format(test_name)
        plt.savefig(filename)
        print('wrote "{0}"'.format(filename))
        plt.close(fig)

    def test1 ():
        domain = np.linspace(-2.0, 2.0, 101)
        mu_v = [0.0, 0.3]
        stddev_v = [1.0, 0.5]
        signal = sum(np.exp(-0.5*((domain-mu)/stddev)**2) for mu,stddev in itertools.izip(mu_v,stddev_v))
        test_fit('test1', domain, signal)

    def test2 ():
        domain = np.linspace(-2.0, 2.0, 101)
        mu_v = [0.0, 0.3]
        stddev_v = [1.0, 0.5]
        signal = sum(np.exp(-0.5*((domain-mu)/stddev)**2) for mu,stddev in itertools.izip(mu_v,stddev_v))
        sample_weights = scipy.signal.hann(len(signal))
        test_fit('test2', domain, signal, sample_weights)

    def test3 ():
        domain = np.linspace(-2.0, 2.0, 101)
        mu_v = [0.0, 0.3]
        stddev_v = [1.0, 0.5]
        signal = sum(np.exp(-0.5*((domain-mu)/stddev)**2) for mu,stddev in itertools.izip(mu_v,stddev_v))
        sample_weights = np.concatenate((np.zeros(30), scipy.signal.hann(41), np.zeros(30)))
        test_fit('test3', domain, signal, sample_weights)

    test0()
    test1()
    test2()
    test3()

