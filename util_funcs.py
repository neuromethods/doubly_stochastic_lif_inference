import numpy as np
import numba
from scipy.special import erfinv
from load_model_params import load_lif
from likelihood_FV import get_params, pISI_fvm_sg
from multiprocessing import pool, cpu_count
from functools import partial


def get_mu_range(sigma, taum=None, min_rate=1e-3, max_rate=200e-3,
                 num_points=100):
    """ Finds a mu discretization for a given sigma, such that the range
    between min_rate and max_rate is covered with num_points.

    :param sigma: float
        White noise standard deviation.
    :param taum: float
        Membrane time constant [ms].
    :param min_rate: float
        Minimal Rate [kHz].
    :param max_rate: float
        Maximal Rate [kHz].
    :param num_points: int
        Number of points used for discretization.
    :return: np.array
        Discretization of mu.
    """

    params = get_params()
    if taum is not None:
        params['tau_m'] = taum
    mu = -1.0
    r_ss = 0.05  # start values (don't have to match)
    while r_ss > min_rate:
        mu -= 0.05
        _, r_ss, _ = EIF_steady_state_numba(params['V_vals'], params['V_r_idx'],
                                               params['tau_m'], params['V_r'],
                                               params['V_T'],
                                               params['Delta_T'], mu, sigma)
        r_ss = r_ss / (1 + r_ss * params['T_ref'])

    mu_min = mu
    mu = -1.0
    r_ss = 0.05  # start values (don't have to match)
    while r_ss < max_rate:
        _, r_ss, _ = EIF_steady_state_numba(params['V_vals'], params['V_r_idx'],
                                               params['tau_m'], params['V_r'],
                                               params['V_T'],
                                               params['Delta_T'], mu, sigma)
        r_ss = r_ss / (1 + r_ss * params['T_ref'])
        mu += 0.05
    mu_max = mu
    mu_range = np.linspace(mu_min, mu_max, num_points)
    return mu_range

def get_mu_range_Poisson(min_rate=1e-3, max_rate=200e-3, num_points=100):
    """ Finds a mu discretization for the Poisson process, such that the range
        between min_rate and max_rate is covered with num_points.

        :param sigma: float
            White noise standard deviation.
        :param taum: float
            Membrane time constant [ms].
        :param min_rate: float
            Minimal Rate [kHz].
        :param max_rate: float
            Maximal Rate [kHz].
        :param num_points: int
            Number of points used for discretization.
        :return: np.array
            Discretization of mu.
        """

    mu_min = np.log(min_rate)
    mu_max = np.log(max_rate)
    mu_range = np.linspace(mu_min, mu_max, num_points)
    return mu_range

@numba.njit
def transform_function(fx,x,x_new):
    """ Shifts a discretized function to another grid, via linear interpolation.

    :param fx: np.array
        Function values at x.
    :param x:  np.array
        Points, where function is evaluated.
    :param x_new: np.array
        Points, where function is required.
    :return:
    """
    fx_new = np.empty(x_new.shape)
    for ix_new, x_new_tmp in enumerate(x_new):
        if x_new_tmp <= x[0]:
            fx_new[ix_new] = fx[0]
        elif x_new_tmp >= x[-1]:
            fx_new[ix_new] = fx[-1]
        else:
            idx, distx = interpolate_x(x_new_tmp, x)
            fx_new[ix_new] = (fx[idx + 1] - fx[idx])*distx + fx[idx]
    return fx_new

@numba.njit
def interpolate_x(xi, rangex):
    """ Interpolates linearly???? TODO

    :param xi:
    :param rangex:
    :return:
    """
    dimx = len(rangex)
    #DETERMINE WEIGHTS FOR X COORDINATE
    if xi <= rangex[0]:
        idx = 0
        distx = 0.0
    elif xi >= rangex[-1]:
        idx = -1
        distx = 0.0
    else:
        for i in xrange(dimx-1):
            if rangex[i] <= xi and xi < rangex[i+1]:
                idx = i
                distx = (xi-rangex[i])/(rangex[i+1]-rangex[i])

    return idx, distx

@numba.njit
def find_isi_idx(FPT_times, ISIs):
    """ Finds the bin indices, that are closest to the time discretized ISI
    distribution.

    :param FPT_times: np.array
        Time discretization of ISI distribution.
    :param ISIs: np.array
        ISIs one needs the indices for.
    :return:  np.array
        Indices of bins where ISIs are closest to.
    """

    ISI_idx = np.empty(ISIs.shape)

    for iisi, isi in enumerate(ISIs):
        ISI_idx[iisi] = int(interpolate_x(isi, FPT_times)[0])

    return  ISI_idx

def gaussian_percentiles(p, mean, std):
    """ Returns quantiles of a univariate Gaussian.

    :param p: np.array
        Quantiles that should be computed (between 0 and 1).
    :param mean: float
        Mean of the Gaussian.
    :param std: float
        Standard deviation of the Gausssian.
    :return: np.array
        Value of quantiles.
    """
    return mean + std * np.sqrt(2.) * erfinv(2. * p - 1.)

@numba.njit
def EIF_steady_state_numba(V_vec, kr, taum, Vr, VT, DeltaT, mu, sigma):
    """ Finds the steady state statistics of a stationary EIF neuron.

    :param V_vec: np.array
        Discretization of membrane potential.
    :param kr: int
        Index of reset membrane potential.
    :param taum: float
        Membrane time constant
    :param Vr: float
        Reset membrane potential.
    :param VT: float
        Parameter for EIF. Can be neglected here.
    :param DeltaT: float
        Has to be 0 for LIF.
    :param mu: float
        Mean white noise input.
    :param sigma: float
        Standard deviation of white noise
    :return: list
        Stationary statistics TODO
    """
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    p_ss = np.zeros(n);  q_ss = np.ones(n);
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-Psi )/taum - mu )
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, kr, -1):
        if not F[k]==0.0:
            p_ss[k-1] = p_ss[k] * A[k] + B[k]
        else:
            p_ss[k-1] = p_ss[k] * A[k] + sig2term_dV
        q_ss[k-1] = 1.0
    for k in xrange(kr, 0, -1):
        p_ss[k-1] = p_ss[k] * A[k]
        q_ss[k-1] = 0.0
    p_ss_sum = np.sum(p_ss)
    r_ss = 1.0/(dV*p_ss_sum)
    p_ss *= r_ss;  q_ss *= r_ss;
    return p_ss, r_ss, q_ss

def get_stationary_stats(mu, sigma):
    """ Returns the mean firing rate and the mean membrane potential for a
    white noise input with certain mean and standard deviation.

    :param mu: float
        Mean of white noise.
    :param sigma: float
        Standard deviation
    :return: list
        Mean firing rate and mean membrane potential.
    """

    tau_m, V_th, V_r, VT, V_lb, DeltaT, Tref, V_vec, kr = load_lif()

    p_ss, r_ss, q_ss = EIF_steady_state_numba(V_vec, kr, tau_m, V_r, VT, DeltaT,
                                           mu, sigma)
    if r_ss == 0:
        r_ss_ref = 0.
        return r_ss_ref, np.nan
    else:
        r_ss_ref = 1./(1./r_ss + Tref)
    p_ss = r_ss_ref * p_ss/r_ss
    dV = V_vec[1]-V_vec[0]
    Vs = V_vec[-1]
    Vmean_ss = dV*np.sum(V_vec*p_ss) + (1-r_ss_ref/r_ss)*(Vs+V_r)/2.

    return r_ss_ref, Vmean_ss

def get_random_parameters(N, sigma):
    """ Samples randomly reasonable parameter values for couplings and offsets.

    :param N: int
        Number of neurons.
    :param sigma: float
        Noise amplitude.
    :return: tuple
        Couplings and offsets.
    """
    mu_bar_range = np.arange(-6., -3.2, .01)
    C_range = np.arange(.1, 1., .01)
    X_MESH, C_MESH = np.meshgrid(mu_bar_range, C_range)
    valid_mu_bars, valid_Cs, valid_idx, rate_p01, rate_p99 = \
        get_valid_range(X_MESH, C_MESH, sigma)
    num_poss_pairs = len(valid_mu_bars)
    param_idx = np.random.randint(0, num_poss_pairs, N)
    Cs = valid_Cs[param_idx]
    mus = valid_mu_bars[param_idx]
    return Cs, mus


def get_valid_range(MU_BAR, C, sigma):
    """ Finds, which (mu,C) pairs result in a process, such that the firing
    rate distribution has a 1% quantile, such that it is not below 1Hz,
    and a 99% quantile, such that it is not above 110Hz.

    :param MU_BAR:  np.array
        Offsets.
    :param C: np.array
        Couplings.
    :param sigma: float
        Standard deviation of white noise.
    :return: list
        Valid offsets, valid couplings, indices of valid pairs, the rates of
        the 1% and 99% quantiles of the rate distribution.
    """
    percentiles_p01 = .01
    mu_p01 = gaussian_percentiles(percentiles_p01, MU_BAR, C)
    percentiles_p99 = .99
    mu_p99 = gaussian_percentiles(percentiles_p99, MU_BAR, C)
    rate_p01 = np.empty(mu_p01.shape)
    rate_p99 = np.empty(mu_p99.shape)

    for i in range(mu_p01.shape[0]):
        for j in range(mu_p01.shape[1]):
            rate_p01[i, j] = get_stationary_stats(mu_p01[i, j], sigma)[0] * 1e3
            rate_p99[i, j] = get_stationary_stats(mu_p99[i, j], sigma)[0] * 1e3

    invalid_simulations = np.where(np.logical_or(rate_p01 < 1., rate_p99 >
                                                 110.))
    rate_p01[invalid_simulations] = np.nan
    rate_p99[invalid_simulations] = np.nan
    valid_idx = np.where(np.logical_not(np.isnan(rate_p01)))
    valid_mus, valid_zetas = MU_BAR[valid_idx], C[valid_idx]

    return valid_mus, valid_zetas, valid_idx, rate_p01, rate_p99

def _pre_calculate_likelihood(sigma, taum=None, gradient=False):
    """ Calculates the ISI distribution for a certain sigma.

    :param sigma: float
        Standard deviation of white noise.
    :param taum:  float
        Membrane time constane [ms]
    :param gradient: Bool
        If true numeraical gradient will be calculated.
    :return: list
        Time discretization, FPT_density (Time discretization x mu
        discretization), mu discretization, value of white noise std.
    """

    params = get_params()
    if taum is not None:
        params['tau_m'] = taum
    mu_range = get_mu_range(sigma, taum=taum)
    FPT_density = np.empty([len(mu_range),len(params['t_grid'])])
    if gradient:
        FPT_density_delta = np.empty([len(mu_range), len(params['t_grid'])])
        delta_mu = 1e-2

    for imu, mu_tmp in enumerate(mu_range):
        mu_array = mu_tmp * np.ones_like(params['t_grid'])
        sigma_array = sigma * np.ones_like(params['t_grid'])
        sp_start = np.array([0])
        fvm_dict = pISI_fvm_sg(mu_array, sigma_array, params, fpt=True,
                    rt=sp_start)

        FPT_density[imu] = fvm_dict['pISI_values']*params['fvm_dt']

        if gradient:
            mu_array = mu_tmp * np.ones_like(params['t_grid']) + delta_mu
            sp_start = np.array([0])
            fvm_dict = pISI_fvm_sg(mu_array, sigma_array, params, fpt=True,
                                   rt=sp_start)
            FPT_density_delta[imu] = fvm_dict['pISI_values'] * params['fvm_dt']
    FPT_times = params['t_grid']
    if gradient:
        dFPT_density = (FPT_density_delta - FPT_density)/delta_mu
        return FPT_times, FPT_density, dFPT_density, mu_range, sigma
    else:
        return FPT_times, FPT_density, mu_range, sigma

def _pre_calculate_Poisson_likelihood():
    """ Calculates the ISI distribution for a Poisson process with rate
    lambda=exp(mu).

    :return: list
    Time discretization, FPT_density (Time discretization x mu
        discretization), mu discretization, 0 (place holder for white noise
        std.)
    """
    params = get_params()
    mu_range = get_mu_range_Poisson()
    rates = np.exp(mu_range)
    FPT_times = params['t_grid']
    FPT_density = rates[:,None]*np.exp(- rates[:,None]*FPT_times[None,
                                                       :]) * params['fvm_dt']
    return FPT_times, FPT_density, mu_range, 0

def pre_calculate_likelihood(sigmas, taum=None, gradient=False, Poisson=False):
    """ Function to precompute ISI distribution for different sigmas. Uses
    parallel processing.

    :param sigmas: np.array
        Sigma values for which the likelihood is computed.
    :param taum: float
        Membrane time constant [ms].
    :param gradient: bool
        Whether numerical gradients should be computed. (Default=False)
    :param Poisson: bool
        Whether Poisson process, instead of LIF likelihood should be
        computed. (Default=True)
    :return:
    """
    ### Note, that for Poisson all sigmas have to be zero.

    num_cpu = int(np.floor(.9*cpu_count()))
    sigma_set = np.unique(sigmas)
    N = len(sigmas)

    if not Poisson:
        if len(sigma_set) > 1:
            f = partial(_pre_calculate_likelihood, taum=taum,
                        gradient=gradient)
            p = pool.Pool(num_cpu)
            FPT_density_results = p.map(f, sigma_set)
        else:
            FPT_density_results = []
            FPT_density_results.append(_pre_calculate_likelihood(sigma_set[
                                                                     0],
                                                                 taum=taum,
                                                          gradient=gradient))

        result_dict = {}
        for result in FPT_density_results:
            result_dict[result[-1]] = result

        if gradient:
            FPT_density_shape, dFPT_density_shape = FPT_density_results[0][
                                                        1].shape,\
                                                    FPT_density_results[0][2].shape
            FPT_density, dFPT_density, mu_ranges = np.empty([N,
                                                  FPT_density_shape[0],
                                                  FPT_density_shape[1]]), \
                                        np.empty([N, dFPT_density_shape[0],
                                                  dFPT_density_shape[1]]),\
                                        np.empty([N, FPT_density_shape[0]])
            for iN in range(N):
                FPT_times = result_dict[sigmas[iN]][0]
                FPT_density[iN] = result_dict[sigmas[iN]][1]
                dFPT_density[iN] = result_dict[sigmas[iN]][2]
                mu_ranges[iN] = result_dict[sigmas[iN]][3]

            return FPT_times, FPT_density, dFPT_density, mu_ranges
        else:
            FPT_density_shape = FPT_density_results[0][1].shape
            FPT_density, mu_ranges = np.empty([N, FPT_density_shape[0],
                                               FPT_density_shape[1]]),\
                                     np.empty([N, FPT_density_shape[0]])
            for iN in range(N):
                FPT_times = result_dict[sigmas[iN]][0]
                FPT_density[iN] = result_dict[sigmas[iN]][1]
                mu_ranges[iN] = result_dict[sigmas[iN]][2]

            return FPT_times, FPT_density, mu_ranges
    elif Poisson:
        FPT_density_results = []
        FPT_density_results.append(_pre_calculate_Poisson_likelihood())
        result_dict = {}
        for result in FPT_density_results:
            result_dict[result[-1]] = result

        FPT_density_shape = FPT_density_results[0][1].shape
        FPT_density, mu_ranges = np.empty([N, FPT_density_shape[0],
                                           FPT_density_shape[1]]), \
                                 np.empty([N, FPT_density_shape[0]])
        for iN in range(N):
            FPT_times = result_dict[sigmas[iN]][0]
            FPT_density[iN] = result_dict[sigmas[iN]][1]
            mu_ranges[iN] = result_dict[sigmas[iN]][2]

        return FPT_times, FPT_density, mu_ranges

def prepare_data(Spikes_pop, FPT_times, FPT_density, mu_ranges, T,
                 sorting_error=0.,
                 valid_spikes=None):
    """ Prepares data, such that it can be used for the fitting procedure.

    :param Spikes_pop: list
        Entries are arrays with spike times for each neuron [ms].
    :param FPT_times: np.array
        Time discretization of ISI density.
    :param FPT_density: np.array
        ISI density for different sigmas.
    :param mu_ranges:
        mu discretizations for different ISI densities (depend on sigma).
    :param T: float
        Recording time [ms].
    :param sorting_error: float
        Fraction of how many errors are expected in the sorting. (Default=0)
    :param valid_spikes: np.array
        Spikes that should be ignored in the observation model. If None all
        spikes are considered. (Default=None)
    :return: list
        All variables that are required for the fitting but do not change.
    """
    N = len(Spikes_pop)
    min_x, max_x, dx = -3.5, 3.5, .05 # Wide enough? Before, it was -5,5
    x_range = np.arange(min_x,max_x,dx)
    px0 = np.exp(-.5*x_range**2)/np.sqrt(2.*np.pi)*dx
    end_ISIs_time = np.array([])
    ISIs = np.array([])
    neuron_ids = np.array([])
    valid_ISIs = np.array([])
    sort_error = np.zeros(N)

    for ineuron in range(N):
        end_ISIs_time = np.concatenate([end_ISIs_time, Spikes_pop[ineuron][1:]])
        ISIs_unit = np.diff(Spikes_pop[ineuron])
        ISIs = np.concatenate([ISIs, ISIs_unit])
        neuron_ids = np.concatenate([neuron_ids,
                                     ineuron*np.ones(len(Spikes_pop[ineuron][1:]))])
        if valid_spikes is None:
            valid_ISIs = np.concatenate([valid_ISIs, np.ones(len(Spikes_pop[
                                                                     ineuron]),
                                        dtype=bool)])
            sort_error[ineuron] = sorting_error
        else:
            val_ISIs_idx = np.array(
                valid_spikes[ineuron][1:]*valid_spikes[ineuron][:-1],
                dtype=bool)
            valid_ISIs = np.concatenate([valid_ISIs, val_ISIs_idx])
            sort_error[ineuron] = np.mean(ISIs_unit[val_ISIs_idx] < 2.)

    sort_ids = np.argsort(end_ISIs_time)
    end_ISIs_time = end_ISIs_time[sort_ids]
    delta_ts = np.diff(np.concatenate([[0], end_ISIs_time, [T]]))
    delta_ts[delta_ts < 1] = 1
    ISIs = ISIs[sort_ids]
    neuron_ids = np.array(neuron_ids[sort_ids],dtype=int)
    valid_ISIs = valid_ISIs[sort_ids]
    ISI_idx = np.array(find_isi_idx(FPT_times, ISIs), dtype=int)
    fixed_args = FPT_density, FPT_times, ISIs, ISI_idx, neuron_ids, \
                 valid_ISIs, delta_ts, x_range, dx, px0, mu_ranges, sort_error
    return fixed_args