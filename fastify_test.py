import sys
sys.path.append('library')

import numpy as np
import symbolic
import sympy
import tensor

if __name__ == '__main__':
    a = symbolic.tensor('a', (3,5,3))
    v = symbolic.tensor('v', (3,))
    p = tensor.contract('i,ijk,k', v, a, v, dtype=object)
    I = np.array([1.0 for _ in range(5)])
    # sin_v = np.vectorize(sympy.sin)
    s = tensor.contract('i,i', I, np.array([sympy.sin(c) for c in p]), dtype=object)
    X = list(a.flat) + list(v.flat)
    ds_dX = np.array([s.diff(x) for x in X])
    f = sum(ds_dx**2 for ds_dx in ds_dX)

    print 'This program is designed to determine which is the best technique for producing fast evaluations from symbolic expressions.'
    print 'f = {0}'.format(f)
    print 'X = {0}'.format(X)

    import sympy.utilities.autowrap

    aw = sympy.utilities.autowrap.autowrap(f, args=X, backend='cython')
    bf = sympy.utilities.autowrap.binary_function('bf', f, args=X, backend='cython')
    uf = sympy.utilities.autowrap.ufuncify(X, f, language='C')
    lf = sympy.lambdify(X, f)

    def run_evalf_subs (F_list):
        return [f.evalf(subs=dict(zip(X,F))) for F in F_list]

    def run_autowrap (F_list):
        return [aw(*F) for F in F_list]

    def run_binary_function (F_list):
        return [bf(*F) for F in F_list]

    def run_ufuncify (F_list):
        return [uf(*F) for F in F_list]

    def run_lambdify (F_list):
        return [lf(*F) for F in F_list]

    def time_function_call (func, *args):
        import time
        start = time.time()
        func(*args)
        end = time.time()
        duration = end - start
        return duration

    def profile_function (func_name, func, F_list):
        duration = time_function_call(func, F_list)
        duration_per_iteration = duration / len(F_list)
        print '{0}: {1} evaluations took {2} s (which is {3} per iteration).'.format(func_name, len(F_list), duration, duration_per_iteration)
        return duration_per_iteration

    run_count = 1000
    F_list = [np.random.randn(len(X)) for _ in range(run_count)]

    test_cases = frozenset([ \
        'run_evalf_subs', \
        'run_autowrap', \
        'run_binary_function', \
        # 'run_ufuncify', \ # This was causing a segfault on my machine
        'run_lambdify' \
    ])
    print 'test_cases = {0}'.format(test_cases)
    results = sorted([(test_case,profile_function(test_case, locals()[test_case], F_list)) for test_case in test_cases], key=lambda x : x[1])
    print 'results (best to worst):'
    for test_case,duration_per_iteration in results:
        print '    {0} : {1} s per iteration.'.format(test_case, duration_per_iteration)

