import numpy as np
import numba
from scipy.special import erf
from util_funcs import transform_function
import time
import multiprocessing
from scipy.optimize import minimize
from scipy.optimize import minimize, minimize_scalar
from multiprocessing import cpu_count, pool
from functools import partial

def compute_mllk_at_point(model_variables, fixed_args):
    proc_params, pop_params = model_variables
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    #pop_params = Js, mus, sigmas
    jump_prob = 1. - np.exp(-gamma_jump * delta_ts)
    upper_edge, lower_edge = edge_correction(x_range, dx, jump_prob, x_bar_jump,
                                             zeta_jump)
    alpha, c = compute_forward_messages_mjp(fixed_args, proc_params,
                                            upper_edge, lower_edge, pop_params)
    try:
        mllk = np.sum(np.log(c))
    except RuntimeWarning:
        return -np.inf, None, None

    return mllk, c, alpha

def simplex_mu_fit_wrapper_mllk(mu, neuron_idx, model_variables, fixed_args):
    proc_params, pop_params = model_variables
    Js, mus, sigmas = pop_params
    mus[neuron_idx] = mu
    pop_params = Js, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def parallel_simplex_mu_fit(neuron_idx, model_variables,
                                    fixed_args):
    optimization_func = partial(simplex_mu_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    opt_res = minimize_scalar(optimization_func,
                              bracket=[-6.5, -4.], bounds=[-8., -3.],
                              method='brent',
                              options={'xtol': 1e-3})
    return opt_res.x

def simplex_J_all_fit_wrapper_mllk(J_all, model_variables, fixed_args):
    print(J_all)
    if J_all < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Js, mus, sigmas = pop_params
    Js[:] = J_all
    pop_params = Js, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def simplex_mu_J_fit_wrapper_mllk(mu_J, neuron_idx, model_variables,
                                  fixed_args):
    mu, J = mu_J[0], mu_J[1]
    if J < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Js, mus, sigmas = pop_params
    mus[neuron_idx] = mu
    Js[neuron_idx] = J
    pop_params = Js, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def parallel_simplex_mu_J_fit(neuron_idx, model_variables,
                              fixed_args):
    optimization_func = partial(simplex_mu_J_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    init_variables = np.array([-4, .5])
    initial_simplex = np.array([[-3., .1],
                                [-7., .5],
                                [-5., .7]])
    opt_res = minimize(optimization_func, x0=init_variables,
                       method='Nelder-Mead', tol=1e-3, options={
            'initial_simplex': initial_simplex})
    return opt_res.x

def simplex_J_fit_wrapper_mllk(J, neuron_idx, model_variables, fixed_args):
    if J < 0:
        return np.inf
    proc_params, pop_params = model_variables
    Js, mus, sigmas = pop_params
    Js[neuron_idx] = J
    pop_params = Js, mus, sigmas
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def parallel_simplex_J_fit(neuron_idx, J_init, model_variables,
                                    fixed_args):
    optimization_func = partial(simplex_J_fit_wrapper_mllk,
                                neuron_idx=neuron_idx,
                                model_variables=model_variables,
                                fixed_args=fixed_args)
    opt_res = minimize_scalar(optimization_func,
                              bracket=[.5 * J_init, 2. * J_init],
                              bounds=[0., 2.],
                              method='brent',
                              options={'xtol': 1e-3})
    return opt_res.x

def simplex_gamma_fit_wrapper_mllk(gamma, model_variables, fixed_args):
    print(gamma)
    if gamma <= 0:
        return np.inf
    proc_params, pop_params = model_variables
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    proc_params = gamma, x_bar_jump, zeta_jump
    model_variables = proc_params, pop_params
    mllk = compute_mllk_at_point(model_variables, fixed_args)[0]
    return -mllk

def optimize_population(N, sigmas, fixed_args, gamma_init=1. / 250.,
                        parallel=True):
    proc_params = gamma_init, 0., 1.
    pop_params = np.zeros(N), -6.5 * np.ones(N), sigmas
    model_variables = proc_params, pop_params

    print('Optimize Mu')

    if parallel:
        #num_cpu = 10
        num_cpu = np.amin([N, int(np.floor(.9*cpu_count()))])
        print(num_cpu)
        p = pool.Pool(num_cpu)
        mu_results = p.map(partial(parallel_simplex_mu_fit,
                            model_variables=model_variables,
                      fixed_args=fixed_args), range(N))
        p.close()
        proc_params, pop_params = model_variables
        Js, mus, sigmas = pop_params
        mus = np.array(mu_results)
        pop_params = Js, mus, sigmas
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
            Js, mus, sigmas = pop_params
            mus[ineuron] = opt_mu
            pop_params = Js, mus, sigmas
            model_variables = proc_params, pop_params

    print('Optimize J/gamma shared')
    converged = False
    mllk_cur = -np.inf
    opt_shared_J = .5
    opt_gamma = 1./500.
    while not converged:
        mllk_old = mllk_cur
        print('Optimize J')
        if parallel:
            #num_cpu = 10
            num_cpu = np.amin([N, int(np.floor(.9 * cpu_count()))])
            p = pool.Pool(num_cpu)
            Js_fit = p.map(partial(parallel_simplex_J_fit,
                                   J_init=opt_shared_J,
                                   model_variables=model_variables,
                                   fixed_args=fixed_args), range(N))
            p.close()
        else:
            Js_fit = np.empty(N)
            for ineuron in range(N):
                print('Neuron %d' % ineuron)
                opt_res = minimize_scalar(simplex_J_fit_wrapper_mllk,
                                          bracket=[.5 * opt_shared_J,
                                                   2. * opt_shared_J],
                                          bounds=[0., 2.],
                                          method='brent', args=(ineuron,
                                                                model_variables,
                                                                fixed_args),
                                          options={'xtol': 1e-3})
                opt_J = opt_res.x
                Js_fit[ineuron] = opt_J

        proc_params, pop_params = model_variables
        Js, mus, sigmas = pop_params
        pop_params = np.array(Js_fit), mus, sigmas
        model_variables = proc_params, pop_params

        print('Optimize tau')

        opt_res = minimize_scalar(simplex_gamma_fit_wrapper_mllk,
                                  bracket=[.5*opt_gamma, 2.*opt_gamma],
                                  bounds=[1./2e3, 1./50.],
                                  method='brent', args=(model_variables,
                                                        fixed_args),
                                  options={'xtol': 1e-3})
        opt_gamma = opt_res.x
        mllk_cur = -opt_res.fun
        proc_params, pop_params = model_variables
        gamma_jump, x_bar_jump, zeta_jump = proc_params
        proc_params = opt_gamma, x_bar_jump, zeta_jump
        model_variables = proc_params, pop_params

        convergence = -(mllk_cur - mllk_old)/mllk_cur
        print(convergence)
        converged = convergence < 1e-3

    print('Optimize mu and J')
    if parallel:
        #num_cpu = 10
        num_cpu = np.amin([N, int(np.floor(.9 * cpu_count()))])
        p = pool.Pool(num_cpu)
        mus_Js_fit = p.map(partial(parallel_simplex_mu_J_fit,
                                   model_variables=model_variables,
                                   fixed_args=fixed_args), range(N))
        p.close()
        mus_fit, Js_fit = np.empty(N), np.empty(N)
        for ineuron in range(N):
            mus_fit[ineuron] = mus_Js_fit[ineuron][0]
            Js_fit[ineuron] = mus_Js_fit[ineuron][1]

    else:
        mus_fit, Js_fit = np.empty(N), np.empty(N)
        for ineuron in range(N):
            print('Neuron %d' % ineuron)
            init_variables = np.array([-4, .5])
            initial_simplex = np.array([[-3., .1],
                                        [-7., .5],
                                        [-5., .7]])
            opt_res = minimize(simplex_mu_J_fit_wrapper_mllk, x0=init_variables,
                               method='Nelder-Mead', args=(ineuron,
                                                           model_variables,
                                                           fixed_args),
                               tol=1e-3, options={
                    'initial_simplex': initial_simplex})
            opt_mu, opt_J = opt_res.x[0], opt_res.x[1]
            mus_fit[ineuron] = opt_mu
            Js_fit[ineuron] = opt_J

    proc_params, pop_params = model_variables
    Js, mus, sigmas = pop_params
    pop_params = np.array(Js_fit), np.array(mus_fit), sigmas
    model_variables = proc_params, pop_params

    return model_variables

def compute_neg_mllk_simplex(variables, model_variables, fixed_args):
    gamma_jump, mu, J = variables
    proc_params, pop_params = model_variables
    proc_params = np.array([gamma_jump, 0., 1.])
    pop_params[0][:] = J
    pop_params[1][:] = mu
    x_bar_jump, zeta_jump = proc_params[1], proc_params[2]
    if gamma_jump > 100e-3 or gamma_jump < .1e-3:
        return np.inf
    if J > 1.5 or J < .0:
        return np.inf

    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_range, sort_error = fixed_args
    #pop_params = Js, mus, sigmas
    jump_prob = 1. - np.exp(-gamma_jump * delta_ts)
    upper_edge, lower_edge = edge_correction(x_range, dx, jump_prob, x_bar_jump,
                                             zeta_jump)
    alpha, c = compute_forward_messages_mjp(fixed_args, proc_params,
                                            upper_edge, lower_edge, pop_params)
    mllk = np.sum(np.log(c))

    return -mllk

def simplex_fit_single_unit(model_variables, fixed_args, initial_simplex=None):

    init_variables = np.array([10e-3, 0., .5])

    if initial_simplex is None:
        initial_simplex = np.array([[1./.1e3, -6.5, .5],
                                    [1./.2e3, -5.5, .1],
                                    [1./1e3, -4.5, 1.],
                                    [1./.25e3, -4.5, .4]])
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
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_range, sort_error = fixed_args

    if gamma_jump < .1e-3:
        gamma_jump = .1e-3
    elif gamma_jump > 100e-3:
        gamma_jump = 100e-3
    if zeta_jump < 1.5 * dx:
        zeta_jump = 1.5 * dx

    proc_params = gamma_jump, x_bar_jump, zeta_jump

    return proc_params

def keep_pop_params_in_range(pop_params, fixed_args):
    Js, mus, sigmas = pop_params
    Js[Js < 0] = 0.
    Js[Js > 1.5] = 1.5

    pop_params = Js, mus, sigmas

    return pop_params

@numba.njit
def compute_forward_messages_mjp(fixed_args, proc_params,
                                 upper_edge, lower_edge, pop_params):
    """ Forward propagation. (Filtering)

        :param FPT_density: np.array [mu_dim x ISI_time_dim]
            Likelihoods for ISIs given specific mu
        :param FPT_times: np.array [ISI_time_dim]
            Discretized time axis
        :param pmu0: np.array [mu_dim]
            Initial distribution of mean input
        :param ISIs: np.array [N_isi]
            All ISIs from spike train
        :param transition_args: list
            Contains all other parameters required
        :param transition_prob_func: function
            Function computing the transition density for the assumed process
        :return: np.array [N_isi x mu_dim],
            Forward probabilities p(mu_i|isi_0:i)
        """
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    sort_error_increment = 1. / float(len(FPT_times))
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    jump_prob = 1. - np.exp(- gamma_jump * delta_ts)
    Js, mus, sigmas = pop_params
    alpha = np.empty((ISIs.shape[0] + 1, px0.shape[0]))
    c = np.empty((ISIs.shape[0] + 1))
    alpha[0] = np.exp(-.5*x_range**2)/np.sqrt(2.*np.pi)*dx
    c[0] = np.sum(alpha[0])
    for spk_idx in range(len(delta_ts) - 1):

        ptrans = jump_process_transition(delta_ts[spk_idx], x_range,
                                         dx, jump_prob[spk_idx],x_bar_jump,
                                         zeta_jump, upper_edge[spk_idx],
                                         lower_edge[spk_idx])
        neuron_id = neuron_idx[spk_idx]
        mu_eff = Js[neuron_id] * x_range + mus[neuron_id]
        mu_range = mu_ranges[neuron_id]
        if valid_ISIs[spk_idx]:
            likel_idx = ISI_idx[spk_idx]
            fpt_dens = (1. - sort_error[neuron_id]) * \
                       FPT_density[neuron_id, :, likel_idx] + \
                       sort_error[neuron_id] * sort_error_increment
            fpt_dens_eff = transform_function(fpt_dens, mu_range, mu_eff)
            alpha[spk_idx + 1] = fpt_dens_eff * np.dot(ptrans, alpha[spk_idx])
        else:
            alpha[spk_idx + 1] = np.dot(ptrans, alpha[spk_idx])
        c[spk_idx + 1] = np.sum(alpha[spk_idx + 1])
        alpha[spk_idx + 1] /= c[spk_idx + 1]

    return alpha, c

def compute_marginals(fixed_args, pop_params, proc_params, alpha, c):
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    jump_prob = 1. - np.exp(- gamma_jump * delta_ts)
    upper_edge, lower_edge = edge_correction(x_range, dx, jump_prob, x_bar_jump,
                                             zeta_jump)
    beta = compute_backward_messages_mjp(fixed_args, proc_params, upper_edge,
                                         lower_edge, pop_params, mu_ranges, c)
    p_mu = alpha*beta
    return p_mu, beta

@numba.njit
def compute_backward_messages_mjp(fixed_args, proc_params, upper_edge,
                                  lower_edge, pop_params, mu_ranges, c):
    """ Backward propagation (Smoothing)

    :param FPT_density: np.array [mu_dim x ISI_time_dim]
        Likelihoods for ISIs given specific mu
    :param FPT_times: np.array [ISI_time_dim]
        Discretized time axis
    :param ISIs: np.array [N_isi]
        All ISIs from spike train
    :param transition_args: list
        Contains all other parameters required
    :param transition_prob_func: function
        Function computing the transition density for the assumed process
    :param c: np.array [N_isi]
        Marginal likelihoods p(mu_i|isi_0:isi)
    :return: np.array [N_isi x mu_dim]
        Backward probabilities ~p(isi_i:N_isi|mu_i)
    """
    FPT_density, FPT_times, ISIs, ISI_idx, neuron_idx, valid_ISIs, delta_ts, \
    x_range, dx, px0, mu_ranges, sort_error = fixed_args
    sort_error_increment = 1. / float(len(FPT_times))
    gamma_jump, x_bar_jump, zeta_jump = proc_params
    Js, mus, sigmas = pop_params
    jump_prob = 1. - np.exp(-gamma_jump * delta_ts)
    beta = np.empty((ISIs.shape[0] + 1, px0.shape[0]))
    beta[-1] = 1.
    N_ISIs = len(ISIs)
    for spk_idx in np.arange(N_ISIs, 0, -1):
        ptrans = jump_process_transition(delta_ts[spk_idx - 1], x_range,
                                         dx, jump_prob[spk_idx - 1],x_bar_jump,
                                         zeta_jump, upper_edge[spk_idx - 1],
                                         lower_edge[spk_idx - 1])
        neuron_id = neuron_idx[spk_idx - 1]
        mu_eff = Js[neuron_id] * x_range + mus[neuron_id]
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
        beta[spk_idx - 1] /= c[spk_idx]#np.sum(beta[spk_idx - 1])

    return beta

@numba.njit
def jump_process_transition(delta_t, x_range, dx, jump_prob, x_bar_jump,
                            zeta_jump, upper_edge, lower_edge):
    """ Calculated the transition probability for jump process

    p(mu_i|mu_i-1) = (1-gamma_mu_jump*isi)*delta(mu_i - mu_i-1) +
                        gamma_mu_jump*isi*N(mu_bar_jump, zeta_jump^2)

    :param delta_t: float
        Current ISI
    :param transition_args: list
        Contains all other parameters required

    :return: np.array [mu_dim x mu_dim]
        Array of transition probabilities
    """

    dim_mu = x_range.shape[0]
    transition_prob = ((1. - jump_prob) * np.identity(dim_mu) + \
                       (jump_prob * np.exp(-.5 * (x_range - x_bar_jump) ** 2
                                          / zeta_jump ** 2) \
                        / np.sqrt(2. * np.pi * zeta_jump ** 2) * dx)).T
    transition_prob[0] += upper_edge
    #    (np.absolute(mu[0] - mu_bar_jump) + dmu / 2.) / (zeta_jump) / np.sqrt(
    #        2.))) / 2.
    transition_prob[-1] += lower_edge
    #    (np.absolute(mu[-1] - mu_bar_jump) + dmu / 2.) / (zeta_jump) / np.sqrt(
    #        2.))) / 2.

    return transition_prob

def edge_correction(x_range, dx, jump_prob, x_bar_jump, zeta_jump):

    upper_edge = jump_prob * (1. - erf((np.absolute(x_range[0] -
                                                    x_bar_jump) + dx / 2.)
                          / (zeta_jump) / np.sqrt(2.))) / 2.
    lower_edge = jump_prob * (1. - erf((np.absolute(x_range[-1] -
                                                    x_bar_jump) + dx / 2.) /
                                       (zeta_jump) / np.sqrt(2.))) / 2.
    return upper_edge, lower_edge