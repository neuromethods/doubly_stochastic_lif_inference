'''necessary functions and classes to compute the ISI density of a stochastic I&F 
neuron with constant parameters, by solving the first passage time problem: a 
Fokker-Planck PDE. This PDE is solved using a finite volume method; for details
see Ladenbauer et al. "Inferring and validating mechanistic models of neural 
microcircuits based on spike-train data" '''

from __future__ import print_function
import numpy as np
from math import exp
from scipy.linalg import solve_banded
from load_model_params import load_lif
# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func

class Grid(object):  # this class implements the voltage discretization

    def __init__(self, V_0=-200., V_1=-40., V_r=-70., N_V=100):
        self.V_0 = V_0
        self.V_1 = V_1
        self.V_r = V_r
        self.N_V = int(N_V)
        # construct the grid object
        self.construct()

    def construct(self):
        self.V_centers = np.linspace(self.V_0, self.V_1, self.N_V)
        # shift V_centers by half of the grid spacing to the left
        # such that the last interface lies exactly on V_l
        self.V_centers -= (self.V_centers[-1] - self.V_centers[-2]) / 2.
        self.dV_centers = np.diff(self.V_centers)
        self.V_interfaces = np.zeros(self.N_V + 1)
        self.V_interfaces[1:-1] = self.V_centers[:-1] + 0.5 * self.dV_centers
        self.V_interfaces[0] = self.V_centers[0] - 0.5 * self.dV_centers[0]
        self.V_interfaces[-1] = self.V_centers[-1] + 0.5 * self.dV_centers[-1]
        self.dV_interfaces = np.diff(self.V_interfaces)
        self.dV = self.V_interfaces[2] - self.V_interfaces[1]
        self.ib = np.argmin(np.abs(self.V_centers - self.V_r))

@njit
def get_v_numba(L, Vi, DT, VT, taum, mu, EIF=True):
    # drift coeffs for EIF/LIF model
    # LIF model
    drift = np.empty(L)
    if not EIF:
        for i in xrange(L):
            drift[i] = mu - Vi[i] / taum
    # EIF model
    else:
        for i in xrange(L):
            drift[i] = (- Vi[i] + DT * exp((Vi[i] - VT) / DT)) / taum + mu
    return drift

@njit
def exp_vdV_D(v, dV, D):  # helper function for diags_A
    return exp(-v * dV / D)

@njit
def matAdt_opt(mat, N, v, D, dV, dt):
    dt_dV = dt / dV

    for i in xrange(1, N - 1):
        if v[i] != 0.0:
            exp_vdV_D1 = exp_vdV_D(v[i], dV, D)
            mat[1, i] = -dt_dV * v[i] * exp_vdV_D1 / (
            1. - exp_vdV_D1)  # diagonal
            mat[2, i - 1] = dt_dV * v[i] / (1. - exp_vdV_D1)  # lower diagonal
        else:
            mat[1, i] = -dt_dV * D / dV  # diagonal
            mat[2, i - 1] = dt_dV * D / dV  # lower diagonal
        if v[i + 1] != 0.0:
            exp_vdV_D2 = exp_vdV_D(v[i + 1], dV, D)
            mat[1, i] -= dt_dV * v[i + 1] / (1. - exp_vdV_D2)  # diagonal
            mat[0, i + 1] = dt_dV * v[i + 1] * exp_vdV_D2 / (
            1. - exp_vdV_D2)  # upper diagonal
        else:
            mat[1, i] -= dt_dV * D / dV  # diagonal
            mat[0, i + 1] = dt_dV * D / dV  # upper diagonal

    # boundary conditions
    if v[1] != 0.0:
        tmp1 = v[1] / (1. - exp_vdV_D(v[1], dV, D))
    else:
        tmp1 = D / dV
    if v[-1] != 0.0:
        tmp2 = v[-1] / (1. - exp_vdV_D(v[-1], dV, D))
    else:
        tmp2 = D / dV
    if v[-2] != 0.0:
        tmp3 = v[-2] / (1. - exp_vdV_D(v[-2], dV, D))
    else:
        tmp3 = D / dV

    mat[1, 0] = -dt_dV * tmp1  # first diagonal
    mat[0, 1] = dt_dV * tmp1 * exp_vdV_D(v[1], dV, D)  # first upper
    mat[2, -2] = dt_dV * tmp3  # last lower
    mat[1, -1] = -dt_dV * (tmp3 * exp_vdV_D(v[-2], dV, D)
                           + tmp2 * (
                           1. + exp_vdV_D(v[-1], dV, D)))  # last diagonal

# initial probability density
def initial_p_distribution(grid, params):
    if params['fvm_v_init'] == 'normal':
        mean_gauss = params['fvm_normal_mean']
        sigma_gauss = params['fvm_normal_sigma']
        p_init = np.exp(-np.power((grid.V_centers - mean_gauss), 2) /
                        (2 * sigma_gauss ** 2))
    elif params['fvm_v_init'] == 'delta':
        delta_peak_index = np.argmin(np.abs(grid.V_centers -
                                            params['fvm_delta_peak']))
        p_init = np.zeros_like(grid.V_centers)
        p_init[delta_peak_index] = 1.
    elif params['fvm_v_init'] == 'uniform':
        # uniform dist on [Vr, Vs]
        p_init = np.zeros_like(grid.V_centers)
        p_init[grid.ib:] = 1.
    else:
        err_mes = ('Initial condition "{}" is not implemented!' + \
                   'See params dict for options.').format(params['fvm_v_init'])
        raise NotImplementedError(err_mes)
    # normalization with respect to the cell widths
    p_init = p_init / np.sum(p_init * grid.dV_interfaces)
    return p_init

@njit
def get_r_numba(v_end, dV, D, p_end):
    # calculation of rate/pISI
    if v_end != 0.0:
        r = v_end * (
        (1. + exp((-v_end * dV) / D)) / (1. - exp((-v_end * dV) / D))) * p_end
    else:
        r = 2 * D / dV * p_end
    return r

def pISI_fvm_sg(mu, sigma, params, fpt=True, rt=list()):
    # solves the Fokker Planck equation (first passage time problem)
    # using the Scharfetter-Gummel finite volume method

    dt = params['fvm_dt']
    T_ref = params['T_ref']
    DT = params['Delta_T']
    VT = params['V_T']
    taum = params['tau_m']

    EIF_model = True if params['neuron_model'] == 'EIF' else False

    # instance of the spatial grid class
    grid = Grid(V_0=params['V_lb'], V_1=params['V_s'], V_r=params['V_r'],
                N_V=params['N_centers_fvm'])

    r = np.zeros_like(mu)

    dV = grid.dV
    Adt = np.zeros((3, grid.N_V))

    rc = 0
    n_rt = len(rt)
    ones_mat = np.ones(grid.N_V)

    # drift coefficients
    v = get_v_numba(grid.N_V + 1, grid.V_interfaces, DT, VT,
                    taum, mu[0], EIF=EIF_model)
    # diffusion coefficient
    D = (sigma[0] ** 2) * 0.5
    # create banded matrix A
    matAdt_opt(Adt, grid.N_V, v, D, dV, dt)
    Adt *= -1.
    Adt[1, :] += ones_mat

    for n in xrange(len(mu)):

        if rc < n_rt and rt[rc] <= n * dt < rt[rc] + dt:
            p = initial_p_distribution(grid, params)
            rc += 1

        if rc - 1 < n_rt and rt[rc - 1] <= n * dt < rt[rc - 1] + T_ref + dt:
            r[n] = 0
        else:

            if n > 0:
                toggle = False
                if mu[n] != mu[n - 1]:
                    # drift coefficients
                    v = get_v_numba(grid.N_V + 1, grid.V_interfaces, DT, VT,
                                    taum, mu[n], EIF=EIF_model)
                    toggle = True
                if sigma[n] != sigma[n - 1]:
                    # diffusion coefficient
                    D = (sigma[n] ** 2) * 0.5
                    toggle = True
                if toggle:
                    # create banded matrix A in each time step
                    matAdt_opt(Adt, grid.N_V, v, D, dV, dt)
                    Adt *= -1.
                    Adt[1, :] += ones_mat

            rhs = p.copy()

            # solve the linear system
            p_new = solve_banded((1, 1), Adt, rhs)

            # compute rate / pISI
            r[n] = get_r_numba(v[-1], dV, D, p_new[-1])

            p = p_new

    results = {'pISI_values': r}
    return results

def get_params():
    """ Gets all the necessary parameters for calculating the ISI density.

    :return: dict
        parameters necessary for the likelihood computation.
    """
    params = dict()
    ## params added
    taum, V_th, V_r, V_T, V_lb, Delta_T, T_ref, V_vec, kr = load_lif()
    ISImax = 1000.
    params['fvm_dt'] = 0.05  # ms, time step for finite volume method
    # 0.1 seems ok, prev. def.: 0.05 ms
    params['tau_m'] = taum
    params['Delta_T'] = Delta_T
    params['T_ref'] = T_ref
    params['neuron_model'] = 'LIF'
    params['integration_method'] = 'implicit'
    params['V_lb'] = V_lb
    params['V_r'] = V_r
    params['V_T'] = V_T
    params['V_s'] = V_th  # Note V_th is V_s
    params['V_vals'] = V_vec
    params['V_r_idx'] = kr

    params['N_centers_fvm'] = 1000
    params['fvm_v_init'] = 'delta'
    params['fvm_delta_peak'] = params['V_r']
    params['t_grid'] = np.arange(0, ISImax, params['fvm_dt'])
    return params