import numpy as np
import os
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view

from src.model import create_connect_mat
from src.sim import sim_determ_lif_recep, gen_sensory_stim, gen_bg_noise

root_dir = r'\Users\Mateo\Documents\BU\ocker group\lif_stochastic_decision'
results_dir = os.path.join(root_dir, 'results')

def produce_prob_decision(mu_0, coh, stim_len, Ne=1600, Ni=400, sparse_prob=None, Jmat=None, save=False):
    E = -70
    v_th = -50
    v_r = -55

    N = Ne + Ni
    f = 0.15 # proportion of selective neurons
    fNe = int(f*Ne)
    fNe2 = int(2*fNe)
    wstrong = 1.7

    tstop = 4
    dt = 0.0001
    tplot = np.arange(0,tstop,dt)

    g_l = {'e': 25, 'i': 20} # E/I leak conductances
    c_m = {'e': 0.5, 'i': 0.2} # E/I membrane capacitances

    g_ext = {'ampa_e': 2.1, 'ampa_i': 1.62} # external input conducatance
    g_rec = {'ampa_e': 0.05, 'nmda_e': 0.165, 'gaba_e': 1.3,
             'ampa_i': 0.04, 'nmda_i': 0.13, 'gaba_i': 1.0} # recurrent conductances
    
    tau_recep = {'ampa': 0.002, 'nmda_rise': 0.002, 'nmda_decay': 0.100, 'gaba': 0.005}

    Vc = {'e': 0, 'i': -70} # E/I target voltages

    if Jmat is None:
        # weight matrix (fully connected)
        Jmat = create_connect_mat(Ne, Ni, f, wstrong, sparse_prob=sparse_prob)

    # stimulation to selective cells
    stim_rates,stim_spikes = gen_sensory_stim(mu_0=mu_0, dt=dt, coh=coh, stim_len=stim_len, N=N, f=f)
    
    # background input noise 
    bg_spks = gen_bg_noise(2400, dt, N, tstop)

    # simulate spiking network
    v, spktimes, spktrains, syn_currents, gating_vars = sim_determ_lif_recep(J=Jmat, E=E, g_l=g_l, c_m=c_m, g_ext=g_ext, g_rec=g_rec, tau_recep=tau_recep, Vc=Vc, EI_ratio=(Ne,Ni), In_sens=stim_spikes, bg_spks=bg_spks, tstop=tstop, dt=dt, v_th=v_th, v_r=v_r)

    # plotting rates
    sum_spks_A = spktrains[:,:fNe].sum(axis=1) # sel for A
    sum_spks_B = spktrains[:,fNe:fNe2].sum(axis=1)

    tstep = int(0.005 / dt) # 5 ms step
    twindow = int(0.05 / dt) # 50 ms window length

    window_spks_A = sliding_window_view(sum_spks_A, window_shape=twindow)
    window_spks_B = sliding_window_view(sum_spks_B, window_shape=twindow)

    firing_rate_A = window_spks_A[::tstep].sum(axis=1) / (fNe) / twindow / dt
    firing_rate_B = window_spks_B[::tstep].sum(axis=1) / (fNe) / twindow / dt

    fr_list = [firing_rate_A, firing_rate_B]

    choice_fr = fr_list[np.sum(firing_rate_A) < np.sum(firing_rate_B)]

    dec_thres = 15
    if np.sum(choice_fr > dec_thres) > 0:
        reaction_time = np.min(np.where(choice_fr > dec_thres)) * (tstop / len(choice_fr)) - 1
    else:
        reaction_time = np.inf
    # reaction_time = np.min(np.where(choice_fr > dec_thres)) * (tstop/len(choice_fr))
    tplot_rt = np.linspace(0,tstop,len(choice_fr))

    print(f"Reaction time: {reaction_time} seconds")

    # plotting
    fig, ax = plt.subplots(2,2)

    # plot all spikes, edit out later
    ax[0,0].plot(spktimes[:,0], spktimes[:,1], 'k|', markersize=0.5)
    ax[0,0].set_yticks([fNe, fNe2, Ne, N])

    # plot sensory input
    ax[0,1].plot(tplot, stim_rates[:,(fNe-10)], color='r')
    ax[0,1].plot(tplot, stim_rates[:,(fNe2-10)], color='g')
    
    # plot spike times of selective populations
    ax[1,0].plot(spktimes[:,0], spktimes[:,1], 'k|', markersize=0.5)
    ax[1,0].vlines(0.0, ymin=0, ymax=fNe, color='r')
    ax[1,0].vlines(0.0, ymin=fNe, ymax=fNe2, color='g')
    ax[1,0].set_ylim((0,fNe2))

    # plot firing rates
    ax[1,1].plot(tplot_rt, firing_rate_A, color='r')
    ax[1,1].plot(tplot_rt, firing_rate_B, color='g')
    ax[1,1].axhline(15, linewidth='0.7', color='k', linestyle='--')
    ax[1,1].set_ylim((0,np.max(choice_fr)+5))


    fig.tight_layout()

    if save:
        coh_text = str(coh)
        coh_text = coh_text.replace('.', '')
        st_text = str(stim_len)
        st_text = st_text.replace('.', '')
        savefile = os.path.join(results_dir, f'prob_decision_m{mu_0}_c{coh_text}_st{st_text}.pdf')
        fig.savefig(savefile)

    # output vars
    lif_sim = {
        'voltage': v,
        'spike_times': spktimes,
        'spike_trains': spktrains,
        'syn_currents': syn_currents,
        'gating_vars': gating_vars
    }

    sens_input = {'rates': stim_rates, 'spikes': stim_spikes, 'background': bg_spks}

    firing_rates = {'A': firing_rate_A, 'B': firing_rate_B}

    return lif_sim, sens_input, firing_rates, reaction_time
