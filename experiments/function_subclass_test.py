import numpy as np
import sympy as sp

class K(sp.Function):
    def __init__ (self, args):
        if len(args.shape) != 2 or args.shape[0] != 2 or args.shape[1] <= 0:
            raise Exception('Expected qp to be a 2-tensor having shape (2,n) for some n > 0')
        super().__init__(self, args)

    @classmethod
    def eval (cls, args):
        print(f'eval({cls}, {args})')
        return np.dot(args[1,:], args[1,:]) / 2

    def fdiff (self, argindex):
        print(f'K.fdiff; self.args = {self.args}, argindex = {argindex}')
        assert 1 <= argindex <= len(self.args)
        return self.args[argindex-1]

def do_stuff ():
    q0,q1,q2 = q = np.array(sp.symbols('q(0:3)'))
    p0,p1,p2 = p = np.array(sp.symbols('p(0:3)'))
    qp = np.vstack((q, p))
    qp_flat = qp.reshape(-1)

    print(f'qp = {qp}')
    print(f'qp_flat = {qp_flat}')
    print(f'K(qp) = {K(qp)}')
    print(f'K(qp).diff(q0) = {K(qp).diff(q0)}')
    print(f'K(qp).diff(p0) = {K(qp).diff(p0)}')
    print(f'K(qp_flat) = {K(qp_flat)}')
    print(f'K(qp_flat).diff(q0) = {K(qp_flat).diff(q0)}')
    print(f'K(qp_flat).diff(p0) = {K(qp_flat).diff(p0)}')

if __name__ == '__main__':
    do_stuff()
