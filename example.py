import generative_model
import model_fit
import util_funcs
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__=='__main__':
    # Set parameters for data generation
    N = 3                   # Number of neurons
    T = 30e3                # Length of simulation [ms]
    sigma = 4.              # White noise amplitude of LIFs
    true_dynamics = 'oup'   # Type of process dynamics: 'oup' or 'mjp'
    time_const_x = 250.     # Time constant of common input process


    # Generates data
    sigmas = sigma*np.ones(N)
    Cs, mus = util_funcs.get_random_parameters(N, sigma)
    print('Example parameters are sampled.')
    pop_params = [Cs, mus, sigmas]
    if true_dynamics is 'mjp':
        color = 'C0'
        jump_rate = 1./time_const_x
        proc_params = [true_dynamics, jump_rate, 0., 1.]
    elif true_dynamics is 'oup':
        color = 'C3'
        proc_params = [true_dynamics, time_const_x, 0., 1.]
    dtsim = 0.05
    Spk_times, x_t, tgrid, rand_vec = generative_model.generate_pop_spikes(
        proc_params, pop_params, dtsim, T)
    print('Spikes are simulated.')

    # Fits the model
    print('Fitting model with MJP assumption.')
    fit_results_mjp = model_fit.fit_model(Spk_times, 'mjp', sigmas, T,
                        get_posterior_process=True)
    print('MJP model fitted!')
    print('Fitting model with OUP assumption.')
    fit_results_oup = model_fit.fit_model(Spk_times, 'oup', sigmas, T,
                        get_posterior_process=True)
    print('OUP model fitted!')

    # Plots results
    fig1 = plt.figure('Figure1', figsize=(8.5,6.75), facecolor=None)
    gs_traces = gridspec.GridSpec(7,1)
    gs_traces.update(left=.07, right=.93, bottom=.4, top=.95)
    gs_params = gridspec.GridSpec(1,5)
    gs_params.update(left=.07, right=.93, bottom=.07, top=.25)

    ax1 = fig1.add_subplot(gs_traces[0,:2])
    ax2 = fig1.add_subplot(gs_traces[2:4,0])
    ax3 = fig1.add_subplot(gs_traces[5:,0])

    time_grid = np.concatenate([np.array([0]),fit_results_mjp['time grid']])
    x_range = fit_results_mjp['x range']
    p_x_mjp = fit_results_mjp['p x']
    p_x_oup = fit_results_oup['p x']

    if N > 1:
        for n_idx in range(N):
            spks = Spk_times[n_idx]
            ax1.plot(spks, n_idx * np.ones(spks.shape),'k.',ms=2)
        ax1.set_ylim([-5., N-.5])
    else:
        ax1.vlines(Spk_times[0], 0, 1, lw=.1)
    ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim([0., T])
    ax1.set_title('Spike data')

    ax2.pcolor(time_grid, x_range, p_x_mjp.T, rasterized=1,
               cmap='afmhot_r')
    ax2.plot(tgrid, x_t, color=color, alpha=.8)
    ax2.set_title('Fit with MJP')
    ax2.set_xlim([0., T])
    ax2.set_xticks([0, .5 * T, T])
    ax2.set_xticklabels([])
    ax2.set_yticks([-2,0,2])
    ax2.set_ylabel('x(t)')
    ax3.pcolor(time_grid, x_range, p_x_oup.T, rasterized=1,
               cmap='afmhot_r')
    ax3.plot(tgrid, x_t, color=color, alpha=.8)
    ax3.set_title('Fit with OUP')
    ax3.set_xlim([0., T])
    ax3.set_yticks([-2,0,2])
    ax3.set_xticks([0,.5*T,T])
    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('x(t)')

    ax4 = fig1.add_subplot(gs_params[0,0])
    ax5 = fig1.add_subplot(gs_params[0,1])
    ax6 = fig1.add_subplot(gs_params[0,2])
    ax7 = fig1.add_subplot(gs_params[0,4])


    ax4.plot(mus, fit_results_mjp['offsets'],'o', color='C0')
    ax4.plot(mus, fit_results_oup['offsets'],'o', color='C3')
    ax4.plot([-6.5,-3],[-6.5,-3],'k')
    ax4.set_title('Offsets [mV]')
    ax4.set_xlabel('True')
    ax4.set_ylabel('Est.')
    ax4.set_xticks([-6,-3])
    ax4.set_yticks([-6,-3])

    ax5.plot(Cs, fit_results_mjp['couplings'],'o', color='C0')
    ax5.plot(Cs, fit_results_oup['couplings'],'o', color='C3')
    ax5.plot([0,1.5],[0,1.5],'k')
    ax5.set_title('Couplings [mV]')
    ax5.set_xlabel('True')
    ax5.set_xticks([0,1])
    ax5.set_yticks([0,1])

    ax6.plot(time_const_x*1e-3, fit_results_mjp['time const x']*1e-3,'o', color='C0')
    ax6.plot(time_const_x*1e-3, fit_results_oup['time const x']*1e-3,'o', color='C3')
    ax6.plot([0,1],[0,1],'k')
    ax6.set_title('Time const. [s]')
    ax6.set_xlabel('True')
    ax6.set_xticks([0,1])
    ax6.set_yticks([0,1])
    llr = fit_results_oup['marginal log likelihood'] - fit_results_mjp['marginal log likelihood']
    ax7.bar([0],[llr], facecolor=color)
    ax7.set_ylim([-np.absolute(llr)-5, np.absolute(llr) + 5])
    ax7.set_xlim([-.5, .5])
    ax7.hlines(0,-.5,.5)
    ax7.set_ylabel('Log. likelihood ratio')

    plt.show()