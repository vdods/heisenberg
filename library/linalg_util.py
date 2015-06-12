def FloatVector (X):
    import numpy
    return numpy.array(X, dtype=float)

def FloatVectorNormSquared (X):
    # return x.transpose().dot(x)
    return sum(x**2 for x in X.flat)

def FloatMatrix (X):
    import numpy
    rows = len(X)
    cols = len(X[0])
    return numpy.ndarray(shape=(rows,cols), dtype=float, buffer=numpy.array(X))

def FloatMatrixNormSquared (X):
    # return x.transpose().dot(x).trace()
    return sum(x**2 for x in X.flat)

def ComplexNormSquared (z):
    # return (x.conjugate()*x).real
    return z.real**2 + z.imag**2

def ComplexVector (Z):
    import numpy
    return numpy.array(Z, dtype=complex)

def ComplexVectorNormSquared (Z):
    # return x.conjugate().transpose().dot(x).real
    return sum(ComplexNormSquared(z) for z in Z.flat)

def ComplexMatrix (Z):
    import numpy
    rows = len(Z)
    cols = len(Z[0])
    return numpy.ndarray(shape=(rows,cols), dtype=complex, buffer=numpy.array(Z))

def ComplexMatrixNormSquared (Z):
    import numpy
    # return numpy.einsum('ij,ij', x.conjugate(), x).real
    return sum(ComplexNormSquared(z) for z in Z.flat)
