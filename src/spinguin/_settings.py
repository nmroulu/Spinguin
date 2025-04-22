class Settings(dict):
    """
    Settings for the spin dynamics simulations.

    Attributes
    ----------
    ZERO_HAMILTONIAN : float
        Hamiltonian is calculated as a sparse matrix. Matrix elements below this threshold
        will be set to zero after constructing the total Hamiltonian.
    ZERO_AUX : float
        Redfield integral in the relaxation superoperator is calculated by exponentiating an
        auxiliary matrix. The matrix exponential is performed using scaling and squaring method
        together with Taylor series. This threshold is used to estimate the convergence of the
        Taylor series and to increase the sparsity within the squaring step.

        NOTE: Consider tightening this threshold when using extremely short correlation times.
    ZERO_RELAXATION : float
        Relaxation superoperator is calculated as a sparse matrix. Matrix elements below this
        threshold will be set to zero after constructing the total relaxation superoperator.
    ZERO_INTERACTION : float
        If the eigenvalues of the interaction tensor, estimated using the 1-norm, are smaller
        than this threshold, the interaction is ignored when calculating the relaxation superoperator.
    ZERO_PROPAGATOR : float
        Matrix exponential is performed using scaling and squaring method together with Taylor
        series. This threshold is used to estimate the convergence of the Taylor series and to
        increase the sparsity within the squaring step, when calculating the time propagator.
    ZERO_PULSE : float
        Calculating the pulse superoperator involves performing a matrix exponential, which is 
        done using scaling and squaring method together with Taylor series. This value is used
        to estimate the convergence of the Taylor series and to increase the sparsity within the
        squaring step.
    ZERO_THERMALIZATION
        Thermalization is done using the Levitt-di Bari method, which involves a matrix exponential.
        This is done using the scaling and squaring method together with Taylor series. This value
        is used to estimate the convergence of the Taylor series and to increase the sparsity within
        the squaring step.

        NOTE: Consider tightening this threshold when using very low magnetic fields.
    ZERO_EQUILIBRIUM
        Constructing the equilibrium state involves a matrix exponential, which is performed using
        the scaling and squaring method together with Taylor series. This value is used to estimate the
        convergence of the Taylor series and to increase the sparsity withing the squaring step.

        NOTE: Consider tightening this threshold when using very low magnetic fields.
    RELATIVE_ERROR : float
        Relative error for the integration in the auxiliary matrix method.
    DENSITY_THRESHOLD : float
        Time propagators are converted from csc_array to numpy.ndarray if the density of the matrix
        exceeds this threshold.
    """
    ZERO_HAMILTONIAN = 1e-12
    ZERO_AUX = 1e-18
    ZERO_RELAXATION = 1e-12
    ZERO_INTERACTION = 1e-9
    ZERO_PROPAGATOR = 1e-18
    ZERO_PULSE = 1e-18
    ZERO_THERMALIZATION = 1e-18
    ZERO_EQUILIBRIUM = 1e-18
    RELATIVE_ERROR = 1e-6
    DENSITY_THRESHOLD = 0.5