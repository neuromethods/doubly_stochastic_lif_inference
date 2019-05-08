import mjp
import oup
import util_funcs
import numpy as np


def fit_model(Spk_times, dynamics, sigmas, T, get_posterior_process=False,
              sorting_error=0):
    """ Main function that fits the doubly-stochastic model to a given list of
    spike times.

    :param Spk_times: list
        List with N entries, where N is the number of neurons. Each entry is
        a np.array with the spike times (in ms) for one neuron in ascending
        order. It is assumed that recording time starts at 0.
    :param dynamics: str
        Determines which slow common process is used. If 'oup' it is an
        Ornstein-Uhlenbeck process, if 'mjp' is is a Markov jump process.
    :param sigmas: list
        List with N entries (floats), where each entry the noise amplitude of
        the n^th neuron. These are not optimized!
    :param T: float
        Length of recording time (in ms).
    :param get_posterior_process: bool
        Whether the posterior over the slow common input should be computed.
        Defaul = False.
    :param sorting_error: float
        Probability that a spike is sorted falsly. In the paper 0.05 was
        used. Default=0.
    :return:
    """
    N = len(Spk_times)
    # Precalculate the ISI likelihood for a stochastic LIF with constant mu, sigma
    # (for parameters see load_model_params.py) for different sigmas
    print('Precalculating the ISI likelihood of a stochastic I&F neuron with constant mu.')
    FPT_times, FPT_density, mu_ranges = util_funcs.pre_calculate_likelihood(
        sigmas)
    # Prepares data and likelihood
    fixed_args = util_funcs.prepare_data(Spk_times, FPT_times, FPT_density,
                              mu_ranges, T, sorting_error=sorting_error)
    print('Data prepared for fitting.')

    print('Fitting the model. Run time depends on data length and number of '
          'neurons.')
    if N > 1:
        # Fitting for population case
        if dynamics is 'mjp':
            # Fitting for the MJP assumption
            # Fits the model parameters (wit heuristic procedure as described
            # in the paper)
            model_variables_fit = mjp.optimize_population(N, sigmas, fixed_args)
            # Computes the likelihood for the optimal parameters
            mllk, c, alpha = mjp.compute_mllk_at_point(
                model_variables_fit, fixed_args)
            if get_posterior_process:
                # computes the posterior process if required
                p_x, beta = mjp.compute_marginals(fixed_args,
                                                   model_variables_fit[1],
                                                   model_variables_fit[0],
                                                   alpha, c)
        elif dynamics is 'oup':
            # Fitting for the OUP assumption
            # Fits the model parameters (wit heuristic procedure as described
            # in the paper)
            model_variables_fit = oup.optimize_population(N, sigmas, fixed_args)
            # Computes the likelihood for the optimal parameters
            mllk, c, alpha = oup.compute_mllk_at_point(
                model_variables_fit, fixed_args)
            if get_posterior_process:
                # computes the posterior process if required
                p_x, beta = oup.compute_marginals(fixed_args,
                                                   model_variables_fit[1],
                                                   model_variables_fit[0],
                                                   alpha, c)
    else:
        # Fitting for the single neuron case
        if dynamics is 'mjp':
            # Fitting under the MJP assumption
            # Finds the optimal parameters with simplex method
            model_variables_fit, mllk = mjp.simplex_fit_single_neuron(sigmas,
                                                          fixed_args)
            if get_posterior_process:
                # computes the posterior process if required
                p_x = mjp.get_marginals(model_variables_fit, fixed_args)
        elif dynamics is 'oup':
            # Fitting under the OUP assumption
            # Finds the optimal parameters with simplex method
            model_variables_fit, mllk = oup.simplex_fit_single_neuron(sigmas,
                                                          fixed_args)
            if get_posterior_process:
                # computes the posterior process if required
                p_x = oup.get_marginals(model_variables_fit, fixed_args)

    # Saves result in dictionary

    x_range = fixed_args[7]
    time_grid = np.cumsum(fixed_args[6])
    proc_params, pop_params = model_variables_fit
    couplings, offsets, sigmas = pop_params
    if dynamics is 'mjp':
        time_const_x = 1. / proc_params[0]
    elif dynamics is 'oup':
        time_const_x = proc_params[0]

    result_dict = {'marginal log likelihood': mllk,
                   'couplings': couplings,
                   'offsets': offsets,
                   'sigmas': sigmas,
                   'time const x': time_const_x}

    if get_posterior_process:
        result_dict['p x'] = p_x
        result_dict['time grid'] = time_grid
        result_dict['x range'] = x_range

    return result_dict



