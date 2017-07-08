
class NBodyProblemContext:
    def __init__ (self, T=100, F=10, N=3, use_constraint=True):
        # Number of dimensions in configuration space (i.e. x,y,z)
        X = 2
        # Number of derivatives in phase space (2, i.e. position and velocity)
        self.D = D = 2
        # Indicates if the constraint should be used or not.
        self.use_constraint = use_constraint

        self.generate_functions(X, N)

        self.fourier_curve_parameterization = FourierCurveParameterization(period=1.0, F=F, T=T, D=D)
        self.riemann_sum_factor = self.fourier_curve_parameterization.period / self.fourier_curve_parameterization.T

        # 2 indicates that there are two coefficients for each (cos,sin) pair
        self.position_velocity_shape      = position_velocity_shape      = (X,D,N)
        self.fourier_coefficients_shape   = fourier_coefficients_shape   = (X,F,2,N)
        self.lagrange_multipliers_shape   = lagrange_multipliers_shape   = (1,)

        self.time_domain_parameter_count      = time_domain_parameter_count      = prod(position_velocity_shape) + prod(lagrange_multipliers_shape)
        self.frequency_domain_parameter_count = frequency_domain_parameter_count = prod(fourier_coefficients_shape) + prod(lagrange_multipliers_shape)

        self.frequency_domain_parameters = frequency_domain_parameters = np.ndarray((frequency_domain_parameter_count,), dtype=float)
        # Define names for the views.  Note that when assigning into the views, the
        # [:] notation is necessary, otherwise el_form_fourier_coefficients_part and
        # el_form_lagrange_multipliers_part will be reassigned to be different references
        # altogether, and will no longer be views into euler_lagrange_form_buffer.
        self.fc,self.lm = fc,lm = self.frequency_domain_views(self.frequency_domain_parameters)

        fc.fill(0.0)
        # Body 0
        fc[0,0,0,0] = 0.0 # constant offset in x
        fc[0,1,0,0] = 1.0 # cos for x
        fc[1,1,1,0] = 1.3 # sin for y
        # Body 1
        fc[0,0,0,1] = -0.0 # constant offset in x
        fc[0,1,0,1] = -0.9 # cos for x
        fc[1,1,1,1] = -0.5 # sin for y
        # Body 2
        fc[0,0,0,2] = 0.0 # constant offset in x
        # fc[0,1,0,2] = -1.0 # cos for x
        # fc[1,1,1,2] = -1.1 # sin for y

        lm.fill(0.0)

        # Defining this here avoids allocation in compute_euler_lagrange_form.
        self.euler_lagrange_form_buffer = euler_lagrange_form_buffer = np.ndarray((frequency_domain_parameter_count,), dtype=float)
        # Define names for the views.  Note that when assigning into the views, the
        # [:] notation is necessary, otherwise el_form_fourier_coefficients_part and
        # el_form_lagrange_multipliers_part will be reassigned to be different references
        # altogether, and will no longer be views into euler_lagrange_form_buffer.
        self.el_form_fc,self.el_form_lm = self.frequency_domain_views(self.euler_lagrange_form_buffer)

    def generate_functions (self, X, N):
        """
        Use sympy to define various functions and automatically compute their derivatives, crunching
        them down to (probably efficient) Python lambda functions.

        X is the dimension of the configuration space.
        """

        assert X > 0
        assert N > 0
        # Dimension of configuration space Q
        self.X = X
        # Number of bodies in gravitational system
        self.N = N

        # Mass of the bodies
        mass = np.linspace(1.0, float(N), N)

        # q is position of each of the bodies
        q = symbolic.tensor('q', (X,N))
        # v is velocity of each of the bodies
        v = symbolic.tensor('v', (X,N))
        # lm is the lagrange multiplier
        lm = symbolic.variable('lm')
        # P is all variables
        self.P = P = np.array(list(q.reshape(q.size, order='C')) + list(v.reshape(v.size, order='C')) + [lm])
        print 'P = {0}'.format(P)

        # U is potential energy
        U = sum(-mass[i]*mass[j] / sympy.sqrt(np.sum(np.square(q[:,i]-q[:,j]))) for i in xrange(N-1) for j in xrange(i+1,N))
        print 'U = {0}'.format(U)
        # U = -1 / sympy.sqrt(np.sum(np.square(q)))
        # K is kinetic energy.  For now assume unit mass for all bodies.
        K = sum(mass[i]*np.sum(np.square(v[:,i])) for i in xrange(N)) / 2
        print 'K = {0}'.format(K)
        # H is total energy (Hamiltonian)
        H = K + U
        # L is the difference in energy (Lagrangian)
        L = K - U
        # H_0 is the constant value which defines the constraint (that the Hamiltonian must equal that at all times)
        self.H_0 = H_0 = -1.8
        # This is the constraint.  The extra division is used to act as a metric on the lagrange multiplier coordinate.
        C = (H - H_0)**2 / 2 / 100

        # DL = symbolic.D(L, P)
        # # DH = symbolic.D(H, P)
        # DC = symbolic.D(C, P)

        # This is the integrand of the action functional
        Lambda_integrand = L #+ lm*C
        # This is the integrand of the first variation of the action
        DLambda_integrand = symbolic.D(Lambda_integrand, P)
        # This is the integrand for the constraint functional
        C_integrand = C

        # Solving the constrained optimization problem by minimizing the norm squared of DLambda.
        # DDLambda_integrand = symbolic.D(DLambda_integrand, P)
        Obj_integrand = (np.sum(np.square(DLambda_integrand))/2)#.simplify()
        DObj_integrand = symbolic.D(Obj_integrand, P)
        # print 'Obj_integrand =', Obj_integrand
        # print ''
        # print 'DObj_integrand =', DObj_integrand
        # print ''

        replacement_d = {'dtype=object':'dtype=float'}

        # self.L = symbolic.lambdify(L, P, replacement_d=replacement_d)
        # self.DL = symbolic.lambdify(DL, P, replacement_d=replacement_d)
        # self.DDL = symbolic.lambdify(DDL, P, replacement_d=replacement_d)
        # self.H = symbolic.lambdify(H, P, replacement_d=replacement_d)
        # self.DH = symbolic.lambdify(DH, P, replacement_d=replacement_d)

        self.H = symbolic.lambdify(H, P, replacement_d=replacement_d)
        self.Lambda_integrand = symbolic.lambdify(Lambda_integrand, P, replacement_d=replacement_d)
        self.DLambda_integrand = symbolic.lambdify(DLambda_integrand, P, replacement_d=replacement_d)
        self.Obj_integrand = symbolic.lambdify(Obj_integrand, P, replacement_d=replacement_d)
        self.DObj_integrand = symbolic.lambdify(DObj_integrand, P, replacement_d=replacement_d)
        self.C_integrand = symbolic.lambdify(C_integrand, P, replacement_d=replacement_d)

    def time_domain_views (self, time_domain_parameters):
        """
        Returns a tuple (qv,lm), where each of the elements are views into:

            qv[d,x,n] : The position and velocity tensor.  The x index indexes the configuration space (i.e. x,y,z axis),
                        the d index indexes the order of derivative (i.e. 0 is position, 1 is velocity), and the n
                        indexes which gravitational body it is.
            lm[:]     : The Lagrange multiplier.

        Note that slice notation must be used to assign to these views, otherwise a new, unrelated local variable will be declared.
        """
        qv_count = prod(self.position_velocity_shape)
        lm_count = prod(self.lagrange_multipliers_shape)
        assert time_domain_parameters.shape == (self.time_domain_parameter_count,)
        qv = time_domain_parameters[:qv_count].reshape(self.position_velocity_shape, order='C')
        lm = time_domain_parameters[qv_count:].reshape(self.lagrange_multipliers_shape, order='C')
        return qv,lm

    def frequency_domain_views (self, frequency_domain_parameters):
        """
        Returns a tuple (fc,lm), where each of the elements are views into:

            fc[x,f,c,n] : The Fourier coefficients of the curve.  The x index indexes the configuration space (i.e. x,y,z axis),
                          the f index denotes the frequency, c indexes which of cos or sin the coefficient is for (0 for cos,
                          1 for sin), and the n indexes which gravitational body it is.
            lm[:]       : The Lagrange multiplier.

        Note that slice notation must be used to assign to these views, otherwise a new, unrelated local variable will be declared.
        """
        fc_count = prod(self.fourier_coefficients_shape)
        lm_count = prod(self.lagrange_multipliers_shape)
        fc = frequency_domain_parameters[:fc_count].reshape(self.fourier_coefficients_shape, order='C')
        lm = frequency_domain_parameters[fc_count:].reshape(self.lagrange_multipliers_shape, order='C')
        return fc,lm

    def curve_at_t (self, t):
        return np.einsum('dfc,xfcn->dxn', self.fourier_curve_parameterization.fourier_tensor[t,:,:,:], self.fc)

    def curve (self):
        return np.einsum('tdfc,xfcn->tdxn', self.fourier_curve_parameterization.fourier_tensor, self.fc)

    def time_domain_variation_pullback_at_t (self, time_domain_parameter_variation, t):
        """Uses the Fourier-transform-parameterization of the curve to pull back a qv-lm vector to be a fc-lm vector."""
        assert time_domain_parameter_variation.shape == (self.time_domain_parameter_count,)
        td_qv,td_lm = self.time_domain_views(time_domain_parameter_variation)

        retval = np.ndarray((self.frequency_domain_parameter_count,), dtype=float)
        fd_fc,fd_lm = self.frequency_domain_views(retval)

        fd_fc[:] = np.einsum('dxn,dfc->xfcn', td_qv, self.fourier_curve_parameterization.fourier_tensor[t,:,:,:])
        fd_lm[:] = td_lm

        return retval

    def time_domain_parameters_at_t (self, t):
        retval = np.ndarray((self.time_domain_parameter_count,), dtype=float)
        td_qv,td_lm = self.time_domain_views(retval)
        td_qv[:] = self.curve_at_t(t)
        td_lm[:] = self.lm
        return retval

    def Lambda (self):
        return self.riemann_sum_factor * sum(self.Lambda_integrand(self.time_domain_parameters_at_t(t)) for t in xrange(self.fourier_curve_parameterization.T))

    def DLambda_at_time (self, t):
        return self.time_domain_variation_pullback_at_t(self.DLambda_integrand(self.time_domain_parameters_at_t(t)), t)

    def DLambda (self, batch_t_v=None):
        if batch_t_v == None:
            batch_t_v = xrange(self.fourier_curve_parameterization.T)
            batch_size = self.fourier_curve_parameterization.T
        else:
            batch_size = len(batch_t_v)

        return (self.fourier_curve_parameterization.period / batch_size) * sum(self.DLambda_at_time(t) for t in batch_t_v)

    def Obj (self):
        return self.riemann_sum_factor * sum(self.Obj_integrand(self.time_domain_parameters_at_t(t)) for t in xrange(self.fourier_curve_parameterization.T))

    def DObj_at_time (self, t):
        return self.time_domain_variation_pullback_at_t(self.DObj_integrand(self.time_domain_parameters_at_t(t)), t)

    def DObj (self, batch_t_v=None):
        if batch_t_v == None:
            batch_t_v = xrange(self.fourier_curve_parameterization.T)
            batch_size = self.fourier_curve_parameterization.T
        else:
            batch_size = len(batch_t_v)

        return (self.fourier_curve_parameterization.period / batch_size) * sum(self.DObj_at_time(t) for t in batch_t_v)

    def C (self):
        return self.riemann_sum_factor * sum(self.C_integrand(self.time_domain_parameters_at_t(t)) for t in xrange(self.fourier_curve_parameterization.T))

