import numpy as np
import numba
from load_model_params import load_lif

@numba.njit
def simulate_EIF_numba(tgrid, V_init, taum, Vth, Vr, VT, DeltaT, Tref,
                       mu_vec, sigma_vec, rand_vec):
    """ Simulates an exponential or leaky integrate and fire (EIF/LIF) neuron
        subject to Gaussian white noise input with (possibly time-varying) mean
        and standard deviation (mu_vec, sigma_vec)

    :param tgrid: np.array
        Time discretization [ms].
    :param V_init: float
        Initial value of membrane potential [mV].
    :param taum: float
        Membrane time constant [ms].
    :param Vth: float
        Spiking threshold [mV].
    :param Vr: float
        Reset potential [mV].
    :param VT: float
        Parameter of EIF (can be neglected).
    :param DeltaT: float
        Has to be 0 for LIF.
    :param Tref: float
        Absolute refractory period [ms].
    :param mu_vec: np.array
        Time depedent mean input.
    :param sigma_vec: np.array
        Standard deviation of white noise.
    :param rand_vec: np.array
        Random numbers for white noise (standard normal).
    :return: list
        Membrane potential, spike times
    """

    dt = tgrid[1] - tgrid[0]
    V = V_init*np.ones(len(tgrid))
    Sp_times_dummy = np.zeros(int(len(tgrid)/10))
    sp_count = int(0)

    sqrt_dt = np.sqrt(dt)
    input_dt = dt*mu_vec + sigma_vec*sqrt_dt*rand_vec

    f1 = -dt/taum
    f2 = dt/taum * DeltaT
    if not f2>0:
        DeltaT = 1.0  # to make sure we don't get errors below

    for i_t in range(1,len(tgrid)):
        V[i_t] = V[i_t-1] + f1*V[i_t-1] + f2*np.exp((V[i_t-1]-VT)/DeltaT) + \
                 input_dt[i_t-1]
#       refr. period
        if sp_count>0 and tgrid[i_t]-Sp_times_dummy[sp_count-1]<Tref:
            V[i_t] = Vr

        if V[i_t]>Vth:
            V[i_t] = Vr
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]


    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]

    return V, Sp_times

def generate_OUP(mu0, tgrid, dtsim, tau_ou):
    """ Generates an Ornstein Uhlenbeck process with a standard normal as
    stationary distribution.

    :param x0: float
        Starting value
    :param T: float
        Length of time course [ms]
    :param Deltat: float
        Time steps for discretization [ms]
    :param tau_ou: float
        Time constant of process [ms].
    :return: np.array
        Vector with the generated process
    """
    num_time_steps = len(tgrid)
    D = np.sqrt(2. / tau_ou)
    dW = np.sqrt(dtsim)*np.random.randn(
        num_time_steps - 1)
    x = np.empty(num_time_steps)
    x[0] = mu0
    for tidx in range(1,num_time_steps):
        x[tidx] = x[tidx - 1] - x[tidx - 1] / tau_ou * dtsim + D*dW[tidx-1]
    return x

def generate_MJP(x0, tgrid, dtsim, gamma_jump):
    """ Generates markov jump process for mean input.

    :param x0: float
        Starting value
    :param T: float
        Length of time course [ms]
    :param Deltat: float
        Time steps for discretization [ms]
    :param gamma_jump: float
        Jump rate [kHz].
    :return: np.array
        Vector with the generated process
    """
    num_time_steps = len(tgrid)
    jumps = np.random.rand(num_time_steps) < gamma_jump * dtsim
    jump_labels = np.cumsum(jumps)
    num_jumps = np.sum(jumps)
    x_jump = np.random.randn(num_jumps + 1)
    x_jump[0] = x0
    x_t = np.empty(num_time_steps)

    for label in range(1, num_jumps + 1):
        label_idx = np.where(np.equal(jump_labels, label))[0]
        x_t[label_idx] = x_jump[label]

    return x_t

def generate_pop_spikes(proc_params, pop_params, dtsim, T, return_V=False):
    """ Generates spikes, according to the generative model.

    :param proc_params:
    :param pop_params: list
        Neuron specific parameters. Couplings, offsets, noise amplitude.
    :param dtsim: float
        Time step of simulation [ms].
    :param T: float
        Length of simulation [ms].
    :param return_V: bool
        Wether membrane potential should be returned. (Default=False)
    :return: list
        List of arrays of spike times, slow common input process,
        time discretization, random numbers
    """

    tgrid = np.arange(0, T + dtsim / 2., dtsim)
    dynamics = proc_params[0]
    Cs, mu_bars, sigmas = pop_params
    N = len(Cs)

    if dynamics == 'oup':
        tau_ou, x_bar_ou, zeta_ou = proc_params[1], proc_params[2], \
                                   proc_params[3]
        x_t = generate_OUP(x_bar_ou, tgrid, dtsim, tau_ou)
    elif dynamics == 'mjp':
        gamma_jump, x_bar_jump, zeta_jump = proc_params[1], proc_params[2], \
                                     proc_params[3]
        x_t = generate_MJP(x_bar_jump, tgrid, dtsim, gamma_jump)


    Spikes_pop = []
    if return_V:
        V_pop = []

    for ineuron in range(N):
        tau_m, V_th, V_r, VT, V_lb, DeltaT, Tref, V_vec, kr = load_lif()
        mu_i = Cs[ineuron]*x_t + mu_bars[ineuron] #+ EL/tau_m
        V0 = V_r
        rand_vec = np.random.randn(len(tgrid))
        sigma_vec = sigmas[ineuron] * np.ones(len(tgrid))
        V, Sp_times = simulate_EIF_numba(tgrid, V0, tau_m, V_th, V_r, VT,
                                         DeltaT, Tref, mu_i, sigma_vec,
                                         rand_vec)

        Spikes_pop.append(Sp_times)
        if return_V:
            V_pop.append(V)

    if return_V:
        return Spikes_pop, x_t, tgrid, rand_vec, V_pop
    else:
        return Spikes_pop, x_t, tgrid, rand_vec




