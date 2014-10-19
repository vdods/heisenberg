def FloatVector (x):
    import numpy
    return numpy.array(x, dtype=float)

def FloatVectorNormSquared (x):
    return x.transpose().dot(x)

def FloatMatrix (x):
    import numpy
    rows = len(x)
    cols = len(x[0])
    return numpy.ndarray(shape=(rows,cols), dtype=float, buffer=numpy.array(x))

def FloatMatrixNormSquared (x):
    return x.transpose().dot(x).trace()

def ComplexNormSquared (x):
    return (x.conjugate()*x).real

def ComplexVector (x):
    import numpy
    return numpy.array(x, dtype=complex)

def ComplexVectorNormSquared (x):
    return x.conjugate().transpose().dot(x).real

def ComplexMatrix (x):
    import numpy
    rows = len(x)
    cols = len(x[0])    
    return numpy.ndarray(shape=(rows,cols), dtype=complex, buffer=numpy.array(x))

def ComplexMatrixNormSquared (x):
    return x.conjugate().transpose().dot(x).trace().real
