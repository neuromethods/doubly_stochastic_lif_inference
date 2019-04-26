import numpy as np
from generative_model import generate_pop_spikes
from mjp import compute_marginals as compute_marginals_jump
from mjp import compute_mllk_at_point as compute_mllk_at_point_jump
from mjp import optimize_population as optimize_population_jump
from util_funcs import pre_calculate_likelihood, prepare_data, get_mu_range, \
    get_valid_range
from oup import optimize_population as optimize_population_ou
import time

true_dynamics = 'oup'
fit_dynamics = 'mjp'
N = 10
dtsim = 0.05
T = 300e3
T_max = 60e3
sigma = 4.
gamma_jump = 2.e-3
sigmas = sigma*np.ones(N)
FPT_times, FPT_density, mu_ranges = pre_calculate_likelihood(sigmas)

mu_bar_range = np.arange(-6.,-3.2,.01)
J_range = np.arange(.1,1.,.01)
X_MESH, J_MESH = np.meshgrid(mu_bar_range, J_range)
valid_mu_bars, valid_Js, valid_idx, rate_p01, rate_p99 = \
    get_valid_range(X_MESH, J_MESH, sigma)
num_poss_pairs = len(valid_mu_bars)
param_idx = np.random.randint(0,num_poss_pairs,N)
Js = valid_Js[param_idx]
mus = valid_mu_bars[param_idx]
sigmas = sigma*np.ones(N)
pop_params = [Js, mus, sigmas]
dynamics = 'jump_proc'
x_bar_jump = 0.
zeta_jump = 1.
proc_params = [dynamics, gamma_jump, x_bar_jump, zeta_jump]
Spikes_pop, x_t, tgrid, rand_vec = generate_pop_spikes(proc_params, pop_params, dtsim, T)
Spikes_pop_sorted = np.sort(np.concatenate(Spikes_pop))
fixed_args = prepare_data(Spikes_pop, FPT_times, FPT_density, mu_ranges, T)
spikes_to_save = np.where(Spikes_pop_sorted <= T_max)[0]

t1 = time.time()
model_variables_fit_jump = optimize_population_jump(N, sigmas, fixed_args)
fit_time_jump = time.time() - t1

mllk_jump, c, alpha = compute_mllk_at_point_jump(model_variables_fit_jump, fixed_args)
proc_params_fit_jump, pop_params_fit_jump = model_variables_fit_jump
p_mu_jump, beta = compute_marginals_jump(fixed_args, pop_params_fit_jump, proc_params_fit_jump, alpha, c)
gamma_fit_jump = proc_params_fit_jump[0]
Js_fit_jump = pop_params_fit_jump[0]
mus_fit_jump = pop_params_fit_jump[1]