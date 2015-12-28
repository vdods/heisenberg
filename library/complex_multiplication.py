import numpy as np

def generate_complex_multiplication_tensor (dtype=float):
    complex_multiplication_tensor = np.zeros((2,2,2), dtype=dtype)
    complex_multiplication_tensor[0,0,0] =  1
    complex_multiplication_tensor[0,1,1] = -1
    complex_multiplication_tensor[1,0,1] =  1
    complex_multiplication_tensor[1,1,0] =  1
    return complex_multiplication_tensor

if __name__ == '__main__':
    import sympy as sp
    import tensor

    complex_multiplication_tensor = generate_complex_multiplication_tensor(dtype=object)

    a,b,c,d = sp.symbols('a,b,c,d')

    product = ((a + sp.I*b) * (c + sp.I*d)).expand()
    fancy_product = tensor.contract('ijk,j,k', complex_multiplication_tensor, np.array([a,b]), np.array([c,d]), dtype=object)
    fancy_product_as_complex = fancy_product[0] + sp.I*fancy_product[1]

    # print product
    # print fancy_product
    # print fancy_product_as_complex
    # print 'difference:', (product-fancy_product_as_complex).simplify()
    assert (product-fancy_product_as_complex).simplify() == 0
    print 'passed test'
