import numpy as np
import numba
from scipy.special import erf
from util_funcs import transform_function
import multiprocessing
from scipy.optimize import minimize, minimize_scalar
from multiprocessing import cpu_count, pool
from functools import partial

def compute_mllk_at_point(model_variables, fixed_args):
    """ Computes the likelihood of the LIF population model for a given set
    of parameters under the assumption that the common input is an Ornstein
    Uhlenbeck process.

    :param model_variables: list
        Contains model specific parameters. First entry
        are proc_params containing the time constant, mean (always 0) and
        standard deviation (always 1) of the common input x(t). The second entry
        contains the pop_params with the couplings (N dim array), offsets (N dim
        array), and the white noise amplitude (N dim array)
    :param fixed_args: list
        List containing parameters, such as data,
        first passage time look up etc. These parameters are not subject of
        optimisation.
    :return: list
        Returns the marginal log likelihood, the summands for each data
        point, and the forward messages (for later computaion of marginals).
    """
    proc_params, pop_params = model_variables

    tau_ou, x_bar_ou, zeta_ou = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, \
    delta_ts, x_range, dx, px0, mu_ranges, sort_error = fixed_args
    nu_ou, var_ou = compute_trans_moments(delta_ts, x_range, tau_ou, x_bar_ou,
                                          zeta_ou)
    upper_edge, lower_edge = edge_correction(x_range, dx, nu_ou, var_ou)
    alpha, c = compute_forward_messages_oup(fixed_args, nu_ou, var_ou,
                                            upper_edge, lower_edge, pop_params)
    try:
        mllk = np.sum(np.log(c))
    except RuntimeWarning:
        return -np.inf, None, None
    return mllk, c, alpha

def simplex_mu_fit_wrapper_mllk(mu, neuron_idx, model_variables, fixed_args):
    """ Method for optimizer to call, when offset of a neuron should be
    optimized.

    :param mu: float
        Offset that is optimized.
    :param neuron_idx: int
        Index of neuron that should be optimized.
    :param model_variables: list
        Containing all model parameters.
    :param fixed_args: list
        All fixed parameters.
    :return: float
        negative marginal log likelihood
    """
    proc_params, pop_params = model_variables
    Cs, mus, sigmas = pop_params
    mus[neuron_idx] = mu
    pop_params = Cs, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def parallel_simplex_mu_fit(neuron_idx, model_variables,
                                    fixed_args):
    """ Optimizes the offset for a specific neuron, given all other parameters.

    :param neuron_idx: int
        Index of neuron whose offset should be optimized.
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: float
        Optimal value of the offset.
    """
    optimization_func = partial(simplex_mu_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    opt_res = minimize_scalar(optimization_func,
                              bracket=[-6.5, -4.], bounds=[-8., -3.],
                              method='brent',
                              options={'xtol': 1e-3})
    return opt_res.x

def simplex_C_all_fit_wrapper_mllk(C_all, model_variables, fixed_args):
    """ Function for optimizing (homogeneous couplings) of a population

    :param C_all: float
        Value of homogeneous couplings.
    :param model_variables: list
        Model parameters.
    :param fixed_args: list
        Containing all the variables of the model (process and population
        parameters).
    :return: float
        Negative marginal log likelihood
    """
    print(C_all)
    if C_all < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Cs, mus, sigmas = pop_params
    Cs[:] = C_all
    pop_params = Cs, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def simplex_C_fit_wrapper_mllk(C, neuron_idx, model_variables, fixed_args):
    """ Function for optimizing a specific coupling.

    :param C: float
        Value of coupling.
    :param neuron_idx: int
        Index of neuron which coupling belongs to.
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: float
        Negative marginal log likelihood
    """
    if C < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Cs, mus, sigmas = pop_params
    Cs[neuron_idx] = C
    pop_params = Cs, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def simplex_mu_C_fit_wrapper_mllk(mu_C, neuron_idx, model_variables,
                                  fixed_args):
    """ Function for joint optimization an offset and a coupling of a
    specific neuron.

    :param mu_C: tuple
        The offset and the coupling.
    :param neuron_idx: int
        The index of the neuron.
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: float
        Negative marginal log likelihood
    """
    mu, C = mu_C[0], mu_C[1]
    if C < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Cs, mus, sigmas = pop_params
    mus[neuron_idx] = mu
    Cs[neuron_idx] = C
    pop_params = Cs, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def parallel_simplex_mu_C_fit(neuron_idx, model_variables,
                              fixed_args):
    """ Optimizes the offset of a neuron and a coupling jointly.

    :param neuron_idx: int
        The index of the neuron.
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: tuple
        Optimal offset and coupling.
    """

    optimization_func = partial(simplex_mu_C_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    init_variables = np.array([-4,.5])
    initial_simplex = np.array([[-3.,.1],
                                   [-7.,.5],
                                   [-5.,.7]])
    opt_res = minimize(optimization_func, x0=init_variables,
             method='Nelder-Mead', tol=1e-3, options={
            'initial_simplex': initial_simplex})
    return opt_res.x

def parallel_simplex_C_fit(neuron_idx, C_init, model_variables,
                           fixed_args):
    """ Optimizes the couplings for a specific neuron.

    :param neuron_idx: int
        The index of the neuron.
    :param C_init: numpy.array
        Initial values of couplings.
    :param model_variables:
    :param fixed_args:
    :return:
    """
    optimization_func = partial(simplex_C_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    opt_res = minimize_scalar(optimization_func,
                              bracket=[.5 * C_init, 2. * C_init],
                              bounds=[0., 2.],
                              method='brent',
                              options={'xtol': 1e-3})
    return opt_res.x

def simplex_tau_fit_wrapper_mllk(tau, model_variables, fixed_args):
    """ Function for optimizing the time constant of the OUP.

    :param tau: float
        Time constant of OUP.
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: float
        Negative marginal log likelihood.
    """
    print(tau)
    if tau <= 0:
        return np.inf
    proc_params, pop_params = model_variables
    tau_ou, x_bar_ou, zeta_ou = proc_params
    proc_params = tau, x_bar_ou, zeta_ou
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def optimize_population(N, sigmas, fixed_args, tau_init=250., parallel=True):
    """ Function for optimizing the population

    :param N: int
        Number of neurons.
    :param sigmas: np.array
        White noise amplitude values (will not be optimized).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :param tau_init: float
        Initial value of process time constant (ms). Default=250.
    :param parallel: bool
        Whether one wants to parallelize the procedure (approximate, but much
        faster) or do updates sequentially (slow, but less approximative).
        Default=True
    :return:
    """
    proc_params = tau_init, 0., 1.
    pop_params = np.zeros(N), -6.5*np.ones(N), sigmas
    model_variables = proc_params, pop_params

    print('Optimize Mu')

    if parallel:
        num_cpu = np.amin([N, int(np.floor(.9*cpu_count()))])
        print(num_cpu)
        p = pool.Pool(num_cpu)
        mu_results = p.map(partial(parallel_simplex_mu_fit,
                            model_variables=model_variables,
                      fixed_args=fixed_args), range(N))
        p.close()
        proc_params, pop_params = model_variables
        Cs, mus, sigmas = pop_params
        mus = np.array(mu_results)
        pop_params = Cs, mus, sigmas
        model_variables = proc_params, pop_params
    else:
        for ineuron in range(N):
            print('Neuron %d' %ineuron)
            opt_res = minimize_scalar(simplex_mu_fit_wrapper_mllk,
                                      bracket=[-6.5,-4.], bounds=[-8.,-3.],
                                      method='brent', args=(ineuron,
                                                            model_variables,
                                                            fixed_args),
                                  options={'xtol': 1e-3})
            opt_mu = opt_res.x
            proc_params, pop_params = model_variables
            Cs, mus, sigmas = pop_params
            mus[ineuron] = opt_mu
            pop_params = Cs, mus, sigmas
            model_variables = proc_params, pop_params

    print('Optimize C/tau shared')
    converged = False
    mllk_cur = -np.inf
    opt_shared_C = .5
    opt_tau = 500.
    while not converged:
        mllk_old = mllk_cur
        print('Optimize C')
        if parallel:
            num_cpu = np.amin([N, int(np.floor(.9 * cpu_count()))])
            p = pool.Pool(num_cpu)
            Cs_fit = p.map(partial(parallel_simplex_C_fit,
                                   C_init=opt_shared_C,
                                   model_variables=model_variables,
                                   fixed_args=fixed_args), range(N))
            p.close()
        else:
            Cs_fit = np.empty(N)
            for ineuron in range(N):
                print('Neuron %d' % ineuron)
                opt_res = minimize_scalar(simplex_C_fit_wrapper_mllk,
                                          bracket=[.5 * opt_shared_C,
                                                   2. * opt_shared_C],
                                          bounds=[0., 2.],
                                          method='brent', args=(ineuron,
                                                                model_variables,
                                                                fixed_args),
                                          options={'xtol': 1e-3})
                opt_C = opt_res.x
                Cs_fit[ineuron] = opt_C

        proc_params, pop_params = model_variables
        Cs, mus, sigmas = pop_params
        pop_params = np.array(Cs_fit), mus, sigmas
        model_variables = proc_params, pop_params

        print('Optimize tau')

        opt_res = minimize_scalar(simplex_tau_fit_wrapper_mllk,
                                  bracket=[.5*opt_tau, 2.*opt_tau],
                                  bounds=[50., 2e3],
                                  method='brent', args=(model_variables,
                                                        fixed_args),
                                  options={'xtol': 1e-3})
        opt_tau = opt_res.x
        mllk_cur = -opt_res.fun
        proc_params, pop_params = model_variables
        tau_ou, x_bar_ou, zeta_ou = proc_params
        tau_ou = opt_tau
        proc_params = tau_ou, x_bar_ou, zeta_ou
        model_variables = proc_params, pop_params

        convergence = -(mllk_cur - mllk_old)/mllk_cur
        print(convergence)
        converged = convergence < 1e-3

    print('Optimize mu and C')
    if parallel:
        #num_cpu = 10
        num_cpu = np.amin([N, int(np.floor(.9 * cpu_count()))])
        p = pool.Pool(num_cpu)
        mus_Cs_fit = p.map(partial(parallel_simplex_mu_C_fit,
                                   model_variables=model_variables,
                                   fixed_args=fixed_args), range(N))
        p.close()
        mus_fit, Cs_fit = np.empty(N), np.empty(N)
        for ineuron in range(N):
            mus_fit[ineuron] = mus_Cs_fit[ineuron][0]
            Cs_fit[ineuron] = mus_Cs_fit[ineuron][1]

    else:
        mus_fit, Cs_fit = np.empty(N), np.empty(N)
        for ineuron in range(N):
            print('Neuron %d' %ineuron)
            init_variables = np.array([-4, .5])
            initial_simplex = np.array([[-3., .1],
                                        [-7., .5],
                                        [-5., .7]])
            opt_res = minimize(simplex_mu_C_fit_wrapper_mllk, x0=init_variables,
                               method='Nelder-Mead', args=(ineuron,
                                                            model_variables,
                                                            fixed_args),
                               tol=1e-3, options={
                    'initial_simplex': initial_simplex})
            opt_mu, opt_C = opt_res.x[0], opt_res.x[1]
            mus_fit[ineuron] = opt_mu
            Cs_fit[ineuron] = opt_C

    proc_params, pop_params = model_variables
    Cs, mus, sigmas = pop_params
    pop_params = np.array(Cs_fit), np.array(mus_fit), sigmas
    model_variables = proc_params, pop_params

    return model_variables


def compute_neg_mllk_simplex(variables, model_variables, fixed_args):
    """ Function for the optimization in the single neuron case.

    :param variables: list
        All parameters that are optimized (time constant, offset, coupling).
    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: float
        Negative marginal log likelihood
    """
    tau_ou, mu, C = variables
    proc_params, pop_params = model_variables
    proc_params = np.array([tau_ou, 0., 1.])
    pop_params[0][:] = C
    pop_params[1][:] = mu
    x_bar_ou, zeta_ou = proc_params[1], proc_params[2]
    if tau_ou > 10e3 or tau_ou < .01e3:
        return np.inf
    if C > 1.5 or C < 0.:
        return np.inf

    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_range, sort_error = fixed_args
    nu_ou, var_ou = compute_trans_moments(delta_ts, x_range, tau_ou, x_bar_ou,
                                          zeta_ou)
    upper_edge, lower_edge = edge_correction(x_range, dx, nu_ou, var_ou)
    alpha, c = compute_forward_messages_oup(fixed_args, nu_ou, var_ou,
                                            upper_edge, lower_edge, pop_params)
    mllk = np.sum(np.log(c))
    return -mllk

def simplex_fit_single_unit(model_variables, fixed_args, initial_simplex=None):
    """ Optimizes a single neuron recording with simplex method.

    :param model_variables: list
        Containing all the variables of the model (process and population
        parameters).
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :param initial_simplex: np.array or None
        Initial simplex for start of optimization procedure. If None default
        simplex is used (Default=None).
    :return: tuple
        Optimal model variables (process and population) and the value of
        marginal log likelihood.
    """

    init_variables = np.array([10e-3, -6.5, .5])
    if initial_simplex is None:
        initial_simplex = np.array([[.1e3, -6.5, .5],
                                    [.2e3, -5.5, .1],
                                    [1e3, -4.5, 1.],
                                    [.25e3, -4.5, .4]])
    res = minimize(compute_neg_mllk_simplex, x0=init_variables,
          args=(model_variables, fixed_args),
             method='Nelder-Mead', tol=1e-1, options={
        'initial_simplex':initial_simplex})

    opt_params = res.x
    opt_mllk = -res.fun
    proc_params, pop_params = model_variables
    proc_params = np.array([opt_params[0], proc_params[1], proc_params[2]])
    pop_params[0][:] = opt_params[2]
    pop_params[1][:] = opt_params[1]
    model_variables = proc_params, pop_params
    return model_variables, opt_mllk

def keep_proc_params_in_range(proc_params, fixed_args):
    """ Function that ensures that process parameters stay in a valid and
    reasonable range.

    :param proc_params: list
        Process parameters.
    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :return: list
        Valid process parameters.
    """
    tau_ou, x_bar_ou, zeta_ou = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_range, sort_error = fixed_args

    if tau_ou > 10e3:
        tau_ou = 10e3
    elif tau_ou < .01e3:
        tau_ou = .01e3
    if zeta_ou < 1.5 * dx:
        zeta_ou = 1.5 * dx

    proc_params = tau_ou, x_bar_ou, zeta_ou

    return proc_params

def keep_pop_params_in_range(pop_params):
    """ Keeps couplings in a valid and reasonable range.

    :param pop_params: list
        Population parameters.
    :return: list
        Valid population parameters.
    """
    Cs, mus, sigmas = pop_params
    Cs[Cs < 0] = 0.
    Cs[Cs > 2] = 2.

    pop_params = Cs, mus, sigmas

    return pop_params

@numba.njit
def compute_forward_messages_oup(fixed_args, nu_ou, var_ou, upper_edge,
                                 lower_edge, pop_params):
    """ Computes the forward messages (filtering).

    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :param nu_ou: np.array
        Mean of the transition density of the OUP.
    :param var_ou: np.array
        Variance of the transition density of the OUP.
    :param upper_edge: np.array
        How much mass would run out on top of the discritization space. Is
        set to the upper bin of x discretization.
    :param lower_edge: p.array
        How much mass would run out below of the discritization space. Is set to
        the lower bin of x discretization.
    :param pop_params: list
         Population parameters.
    :return: tuple
        The forward messages, and the normalzation from which the marginal
        log likelihood is computed.
    """
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    sort_error_increment = 1./float(len(FPT_times))
    Cs, mus, sigmas = pop_params
    alpha = np.empty((ISIs.shape[0] + 1, px0.shape[0]))
    c = np.empty((ISIs.shape[0] + 1))
    alpha[0] = np.exp(-.5 * x_range ** 2) / np.sqrt(2. * np.pi) * dx
    c[0] = np.sum(alpha[0])

    for spk_idx in range(len(delta_ts) - 1):
        ptrans = ou_process_transition(x_range, dx,
                                       nu_ou[spk_idx],
                                       var_ou[spk_idx],
                                       upper_edge[spk_idx], lower_edge[
                                           spk_idx])
        neuron_id = neuron_idx[spk_idx]
        mu_eff = Cs[neuron_id] * x_range + mus[neuron_id]
        mu_range = mu_ranges[neuron_id]
        if valid_ISIs[spk_idx]:
            likel_idx = ISI_idx[spk_idx]
            fpt_dens = (1. - sort_error[neuron_id]) * \
                       FPT_density[neuron_id,:,likel_idx] + \
                       sort_error[neuron_id]*sort_error_increment
            fpt_dens_eff = transform_function(fpt_dens, mu_range, mu_eff)
            alpha[spk_idx + 1] = fpt_dens_eff * np.dot(ptrans, alpha[spk_idx])
        else:
            alpha[spk_idx + 1] = np.dot(ptrans, alpha[spk_idx])

        c[spk_idx + 1] = np.sum(alpha[spk_idx + 1])
        alpha[spk_idx + 1] /= c[spk_idx + 1]

    return alpha, c

def compute_marginals(fixed_args, pop_params, proc_params, alpha, c):
    """ Computes the marginals of the common process x(t).

    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :param pop_params: list
         Population parameters.
    :param proc_params: list
        Process parameters.
    :param alpha: np.array
        Forward messages.
    :param c: np.array
        Normalization constant of filtering distributions.
    :return: tuple
        The marginal distributions over the process and the backward messages.
    """
    tau_ou, x_bar_ou, zeta_ou = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    nu_ou, var_ou = compute_trans_moments(delta_ts, x_range, tau_ou, x_bar_ou,
                                          zeta_ou)
    upper_edge, lower_edge = edge_correction(x_range, dx, nu_ou, var_ou)
    beta = compute_backward_messages_oup(fixed_args, nu_ou, var_ou, upper_edge,
                                         lower_edge, pop_params, c)
    p_mu = alpha*beta
    return p_mu, beta

@numba.njit
def compute_backward_messages_oup(fixed_args, nu_ou, var_ou, upper_edge,
                                  lower_edge, pop_params, c):
    """ Computes the backwards meassages (smoothing).

    :param fixed_args: list
        All fixed parameters of the optimization procedure.
    :param nu_ou: np.array
        Mean of the transition density of the OUP.
    :param var_ou: np.array
        Variance of the transition density of the OUP.
    :param upper_edge: np.array
        How much mass would run out on top of the discritization space. Is
        set to the upper bin of x discretization.
    :param lower_edge: p.array
        How much mass would run out below of the discritization space. Is set to
        the lower bin of x discretization.
    :param pop_params: list
         Population parameters.
    :param c: np.array
        The normalization constants from the filtering distribution.
    :return: np.array
        Backward messages.
    """

    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    sort_error_increment = 1. / float(len(FPT_times))
    Cs, mus, sigmas = pop_params
    beta = np.empty((ISIs.shape[0] + 1, px0.shape[0]))
    beta[-1] = 1.
    N_ISIs = len(ISIs)
    for spk_idx in np.arange(N_ISIs, 0, -1):
        ptrans = ou_process_transition(x_range, dx,
                                       nu_ou[spk_idx - 1], var_ou[spk_idx - 1],
                                       upper_edge[spk_idx - 1], lower_edge[
                                           spk_idx - 1])
        neuron_id = neuron_idx[spk_idx - 1]
        mu_eff = Cs[neuron_id] * x_range + mus[neuron_id]
        mu_range = mu_ranges[neuron_id]
        if valid_ISIs[spk_idx - 1]:
            likel_idx = ISI_idx[spk_idx - 1]
            fpt_dens = (1. - sort_error[neuron_id]) * \
                       FPT_density[neuron_id, :, likel_idx] + \
                       sort_error[neuron_id] * sort_error_increment
            fpt_dens_eff = transform_function(fpt_dens, mu_range, mu_eff)
            beta[spk_idx - 1] = np.dot(ptrans.T, fpt_dens_eff * beta[spk_idx])
        else:
            beta[spk_idx - 1] = np.dot(ptrans.T, beta[spk_idx])
        beta[spk_idx - 1] /= c[spk_idx]

    return beta

@numba.njit
def ou_process_transition(x_range, dx, nu_ou, var_ou, upper_edge,
                          lower_edge):
    """ Computes the (discretized) transition probability for the OUP.

    :param x_range: np.array
        Discretization of the common process x.
    :param dx: float
        Bin size of discretization.
    :param nu_ou: float
        Mean of transition probability.
    :param var_ou: float
        Variance of transition probability.
    :param upper_edge: np.array
        Probability mass that would run out at the upper end of discretization.
    :param lower_edge: np.array
        Probability mass that would run out at the lower end of discretization.
    :return: np.array
        Transition matrix.
    """
    diffx = np.empty((x_range.shape[0], x_range.shape[0]))
    for ix, x_tmp in enumerate(x_range):
        diffx[ix] = nu_ou - x_tmp
    transition_prob = np.exp(-.5 * (diffx) ** 2 / var_ou) / np.sqrt(
            2. * np.pi * var_ou) * dx

    transition_prob[0] += upper_edge
    transition_prob[-1] += lower_edge

    transition_prob /= np.sum(transition_prob, axis=0)
    return transition_prob

def compute_trans_moments(delta_ts, x_range, tau_ou, x_bar_ou, zeta_ou):
    """ Precomputes the moments of the transition density.

    :param delta_ts: np.array
        Difference of event based time discretization.
    :param x_range: np.array
        Discretization of common process x.
    :param tau_ou: float
        Time constant of OUP.
    :param x_bar_ou: float.
        Depreciated, always 0.
    :param zeta_ou: float
        Depreciated, always 1.
    :return: tuple
        Conditional mean and variances for the transition density.
    """
    nu_ou = np.empty((delta_ts.shape[0], x_range.shape[0]))
    var_ou = np.empty((delta_ts.shape[0]))

    for idelta_t, delta_t in enumerate(delta_ts):
        nu_ou[idelta_t] = x_bar_ou + (x_range - x_bar_ou) * np.exp(-delta_t /
                                                                   tau_ou)
        var_ou[idelta_t] = zeta_ou ** 2 * (1. - np.exp(-2. * delta_t / tau_ou))

    return nu_ou, var_ou

def edge_correction(x_range, dx, nu_ou, var_ou):
    """ Computes the probability mass that would flow out of the discretization.

    :param x_range: np.array
        Discretization of common process x.
    :param dx: float
        Bin size of discretization.
    :param nu_ou: float
        Mean of transition probability.
    :param var_ou: float
        Variance of transition probability.
    :return: tuple
        Arrays with mass flowing out at upper and lower edge of discretization.
    """
    upper_edge = (1. - erf((np.absolute((x_range[0] - nu_ou)) + dx / 2.) /
                           np.sqrt(var_ou[:,np.newaxis]) / np.sqrt(2.))) / 2.
    lower_edge = (1. - erf((np.absolute(
        (x_range[-1] - nu_ou) + dx / 2.) / np.sqrt(var_ou[:, np.newaxis])) / np.sqrt(
        2.))) / 2.

    return upper_edge, lower_edge