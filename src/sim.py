import numpy as np
from src.model import intensity

def sim_pif(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1):

    Nt = int(tstop / dt)
    
    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        v[t] = v[t-1] + dt*E + np.dot(J, n) - n*v[t-1]

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        if lam > 1/dt:
            lam = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t, i])
            
    return v, spktimes


def sim_lif_pop(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0, v0=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    v[0] = v0
    
    n = np.zeros(N,)
    spkind = []
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n) #- n*(v[t-1]-v_r)
        v[t, spkind] = v_r

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_pop_exact_reset(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0, v0=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    v[0] = v0
    
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n)
        v[t, n.astype('int')] = v_r

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_pop_fully_connected(J, E, N=1000, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=0, Estim=0):

    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N))
    v[0] = np.random.randn(N) / np.sqrt(N)
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):

        if t < Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + J*np.sum(n) - n*(v[t-1]-v_r)

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        # lam[lam > 1/dt] = 1/dt

        # n = np.random.binomial(n=1, p=dt*lam)
        n = np.random.poisson(lam=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_perturbation(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, perturb_start=None, perturb_len=10, perturb_amp=1.5, perturb_ind=None):

    '''
    Simulate an LIF network with a perturbation to E.
    If perturb_start=None, at times tstop/4 a postive perturbation is applied, and at 3/4 tstop a negative perturbation.

    J: connectivity matrix, NxN
    E: resting potential
    '''

    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    print(E0.shape)

    if perturb_ind is None:
        perturb_ind = range(N)

    if perturb_start is None:
        t_start_perturb1 = Nt//4
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)
        
        t_start_perturb2 = 3*Nt//4
        t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)
    
    else:
        t_start_perturb1 = int(perturb_start / dt)
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)

        t_start_perturb2 = Nt+1
        t_end_perturb2 = Nt+1

    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)
    n = np.zeros(N,)
    spkind = []

    spktimes = []

    E = E0.copy()

    for t in range(1, Nt):

        if (t >= t_start_perturb1) and (t < t_end_perturb1):
            E[perturb_ind] = E0[perturb_ind] + perturb_amp
        elif (t >= t_start_perturb2) and (t < t_end_perturb2):
            E[perturb_ind] = E0[perturb_ind] - perturb_amp
        else:
            E[perturb_ind] = E0[perturb_ind]

        # v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + J.dot(n)
        v[t] = v[t-1] + dt*(-v[t-1] + E) + J.dot(n)
        v[t, spkind] = v_r # reset

        lam = intensity(v[t], B=B, v_th=v_th, p=p) # IF as a function of voltage
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam) # probabilistic spike based on IF

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i]) # spiking mechanism
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_perturbation_x(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, perturb_amp=1.5, perturb_ind=None):

    '''
    simulate an LIF network. the resting potential E undergoes a series of step perturbations, evenly spaced in time.
    
    J: connectivity matrix, NxN
    E: resting potential
    perturb_amp: list of step amplitudes
    perturb_ind: indices of neurons that experience the perturbation
    '''

    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1

    if len(np.shape(perturb_amp)) == 0:
        Nperturb = 1
    else:
        Nperturb = len(perturb_amp)

    perturb_len = Nt // (Nperturb + 1)

    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    if perturb_ind is None:
        perturb_ind = range(N)

    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)
    n = np.zeros(N,)
    spktimes = []

    for t in range(1, Nt):
        
        E = E0.copy()
        if t > perturb_len:
            E[perturb_ind] += perturb_amp[(t - perturb_len) // perturb_len]

        # v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + J.dot(n)
        v[t] = v[t-1] + dt*(-v[t-1] + E) + J.dot(n)
        v[t, n.astype('int')] = v_r

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes


def sim_lif_time_dep_perturbation(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, E_stim=None, perturb_ind=None):

    '''
    Simulate an LIF network with a perturbation to E.
    If perturb_start=None, at times tstop/4 a postive perturbation is applied, and at 3/4 tstop a negative perturbation.

    J: connectivity matrix, NxN
    E: resting potential
    '''

    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    if E_stim is None:
        E_stim = np.zeros((Nt,))

    if len(E_stim) != Nt:
        raise Exception('Mismatched sim time and Estim, {} and {}'.format(Nt, len(E_stim)))


    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)
    n = np.zeros(N,)
    spkind = []

    spktimes = []

    E = E0.copy()

    for t in range(1, Nt):

        E[perturb_ind] = E0[perturb_ind] + E_stim[t]

        # v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + J.dot(n) # approximate reset
        v[t] = v[t-1] + dt*(-v[t-1] + E) + J.dot(n)
        v[t, spkind] = v_r # reset

        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            spktimes.append([t*dt, i])
            
    spktimes = np.array(spktimes)
    return v, spktimes

def sim_lif_receptor_perturbation(J, E, g_l, c_m, g_rec, tau_recep, Vc, EI_ratio, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, perturb_start=None, perturb_len=10, perturb_amp=1.5, perturb_ind=None):
    """
    Simulate an LIF network with receptor-specific currents with a perturbation to E.
    If perturb_start=None, at times tstop/4 a postive perturbation is applied, and at 3/4 tstop a negative perturbation.

    J: connectivity matrix, NxN
    E: resting potential
    tau_mem: time constants for each neuron, N
    tau_rec: time constants for each receptor, dict
    g_rec: conductances for each receptor
    Vc: leak voltages for each cell type
    """
    Nt = int(tstop / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception('Need either a scalar or length N input E')

    print(E0.shape)

    # E-I ratio
    Ne,Ni = EI_ratio

    if perturb_ind is None:
        perturb_ind = range(N)

    if perturb_start is None:
        t_start_perturb1 = Nt//4
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)
        
        t_start_perturb2 = 3*Nt//4
        t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)
    
    else:
        t_start_perturb1 = int(perturb_start / dt)
        t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)

        t_start_perturb2 = Nt+1
        t_end_perturb2 = Nt+1

    v = np.zeros((Nt,N))
    v[0] = np.random.rand(N,)

    n = np.zeros(N,) # spikes
    spkind = []
    spktimes = []

    C = np.zeros(N,)
    C[:Ne] = c_m["e"]
    C[Ne:] = c_m["i"]

    g_L = np.zeros(N,)
    g_L[:Ne] = g_l["e"]
    g_L[Ne:] = g_l["i"]

    I_syn = np.zeros((Nt,N))

    I_ampa = np.zeros((Nt,N))
    I_nmda = np.zeros((Nt,N))
    I_gaba = np.zeros((Nt,N))

    g_ampa = np.zeros(N,)
    g_ampa[:Ne] = g_rec['ampa_e']
    g_ampa[Ne:] = g_rec['ampa_i']

    g_nmda = np.zeros(N,)
    g_nmda[:Ne] = g_rec['nmda_e']
    g_nmda[Ne:] = g_rec['nmda_i']

    g_gaba = np.zeros(N,)
    g_gaba[:Ne] = g_rec['gaba_e']
    g_gaba[Ne:] = g_rec['gaba_i']

    s_ampa = np.zeros((Nt,Ne))
    s_ampa[0] = np.random.rand(Ne,)

    s_nmda = np.zeros((Nt,Ne))
    s_nmda[0] = np.random.rand(Ne,)

    s_gaba = np.zeros((Nt,Ni))
    s_gaba[0] = np.random.rand(Ni,)

    x = np.zeros((Nt,Ne))
    x[0] = np.random.rand(Ne,)

    tau_ampa = tau_recep['ampa']
    tau_nmda_rise = tau_recep['nmda_rise']
    tau_nmda_decay = tau_recep['nmda_decay']
    tau_gaba = tau_recep['gaba']

    Ve = Vc['e']
    Vi = Vc['i']

    E = E0.copy()


    for t in range(1, Nt):

        if (t >= t_start_perturb1) and (t < t_end_perturb1):
            E[perturb_ind] = E0[perturb_ind] + perturb_amp
        elif (t >= t_start_perturb2) and (t < t_end_perturb2):
            E[perturb_ind] = E0[perturb_ind] - perturb_amp
        else:
            E[perturb_ind] = E0[perturb_ind]

        # v[t] = v[t-1] + dt*(-v[t-1] + E) - n*(v[t-1]-v_r) + J.dot(n)
        #v[t] = v[t-1] + (dt/tau_mem)*(-v[t-1] + E) + J.dot(n)
        I_syn[t-1] = I_ampa[t-1] + I_nmda[t-1] + I_gaba[t-1]
        v[t] = v[t-1] + (dt/C)*(-g_L*(v[t-1] - E) - I_syn[t-1])
        v[t, spkind] = v_r # reset

        lam = intensity(v[t], B=B, v_th=v_th, p=p) # IF as a function of voltage
        lam[lam > 1/dt] = 1/dt

        if np.sum(np.isnan(lam)) > 0:
            print("NaNs detected")
            
        n = np.random.binomial(n=1, p=dt*lam) # probabilistic spike vector based on IF

        spkind = np.where(n > 0)[0] # which neurons spiked
        for i in spkind:
            spktimes.append([t*dt, i]) # spiking mechanism

        I_ampa[t] = g_ampa * (v[t] - Ve) * (J[:,:Ne] @ s_ampa[t-1]) # incoming ampa currents
        I_nmda[t] = g_nmda * (1 + np.exp((-0.062 * v[t]) / 3.57)) * (J[:,:Ne] @ s_nmda[t-1]) # incoming nmda currents
        I_gaba[t] = g_gaba * (v[t] - Vi) * (J[:,Ne:] @ s_gaba[t-1]) # incoming gaba currents

        s_ampa[t] = s_ampa[t-1] - dt*((1/tau_ampa)*s_ampa[t-1]) + n[:Ne]# spikes here
        s_nmda[t] = s_nmda[t-1] - dt*((1/tau_nmda_decay)*s_nmda[t-1] + 0.5*x[t-1]*(1-s_nmda[t-1]))
        x[t] = x[t-1] - dt*((1/tau_nmda_rise)*x[t-1]) + n[:Ne] # spikes here
        s_gaba[t] = s_gaba[t-1] - dt*((1/tau_gaba)*s_gaba[t-1]) + n[Ne:] # spikes here
            
    spktimes = np.array(spktimes)

    gating_vars = {
        "ampa": s_ampa,
        "nmda": s_nmda,
        "gaba": s_gaba,
    }

    syn_currents = {
        "total": I_syn,
        "ampa": I_ampa,
        "nmda": I_nmda,
        "gaba": I_gaba
    }

    return v, spktimes, syn_currents, gating_vars

def gen_bg_noise(rate, dt, N, sim_len):
    Nt = int(sim_len / dt)
    spks = (np.random.rand(Nt, N) < (rate*dt))*1

    return spks

def sim_determ_lif_recep(J, E, g_l, c_m, g_ext, g_rec, tau_recep, Vc, EI_ratio, In_sens, bg_spks, tstop=100, dt=.01, v_th=1, v_r=0, perturb_start=None, perturb_len=10, perturb_amp=1.5, perturb_ind=None):
    """
    Simulate deterministic LIF network with receptor-specific currents
    """
    Nt = int(tstop / dt)

    if len(J.shape) > 1:
        N = J.shape[0]
    else:
        N = 1

    if len(np.shape(E)) == 0:
        E0 = E * np.ones(N,)
    elif len(E) == N:
        E0 = np.array(E)
    else:
        raise Exception("Need either a scalar or length N input E")

    print(f"Simulating network of {E0.size} neurons")

    Ne,Ni = EI_ratio

    v = np.zeros((Nt, N))
    v[0] = E0

    spktrains = np.zeros((Nt,N))
    spktimes = []

    C = np.zeros(N,)
    C[:Ne] = c_m["e"] # excitatory membrane capacitance
    C[Ne:] = c_m["i"] # inhibitory membrane capacitance

    g_L = np.zeros(N,)
    g_L[:Ne] = g_l["e"] # excitatory leak conductance
    g_L[Ne:] = g_l["i"] # inhibitory leak conductance

    # incoming synaptic current into each neuron
    I_syn = np.zeros((Nt,N)) 

    I_ext_ampa = np.zeros((Nt,N))
    I_ampa = np.zeros((Nt,N))
    I_nmda = np.zeros((Nt,N))
    I_gaba = np.zeros((Nt,N))

    # external ampa conductances
    g_ext_ampa = np.zeros(N,)
    g_ext_ampa[:Ne] = g_ext['ampa_e']
    g_ext_ampa[Ne:] = g_ext['ampa_i']

    # recurrent receptor conductances into each neuron
    g_ampa = np.zeros(N,)
    g_ampa[:Ne] = g_rec['ampa_e']
    g_ampa[Ne:] = g_rec['ampa_i']

    g_nmda = np.zeros(N,)
    g_nmda[:Ne] = g_rec['nmda_e']
    g_nmda[Ne:] = g_rec['nmda_i']

    g_gaba = np.zeros(N,)
    g_gaba[:Ne] = g_rec['gaba_e']
    g_gaba[Ne:] = g_rec['gaba_i']

    # gating variables
    s_ext_ampa = np.zeros((Nt,N))
    s_ampa = np.zeros((Nt,Ne))
    s_nmda = np.zeros((Nt,Ne))
    s_gaba = np.zeros((Nt,Ni))
    
    x = np.zeros((Nt,Ne))

    # time constants
    tau_ampa = tau_recep['ampa']
    tau_nmda_rise = tau_recep['nmda_rise']
    tau_nmda_decay = tau_recep['nmda_decay']
    tau_gaba = tau_recep['gaba']

    Ve = Vc['e']
    Vi = Vc['i']

    E = E0.copy()

    for t in range(1, Nt):
        # forward euler loop

        # voltage update
        v[t] = v[t-1] + (dt / C) * (-g_L * (v[t-1] - E) - I_syn[t-1])

        # check spikes, don't update voltage yet
        spkind, = np.where(v[t] >= v_th)
        for i in spkind:
            spktimes.append([t*dt, i]) # spiking mechanism
        n = np.zeros(N,) # which neurons spiked
        n[spkind] = 1
        spktrains[t,:] = n

        # gating variables
        s_ext_ampa[t] = s_ext_ampa[t-1] + dt*((-1/tau_ampa)*s_ext_ampa[t-1]) + bg_spks[t-1] + In_sens[t-1]
        s_ampa[t] = s_ampa[t-1] + dt*((-1/tau_ampa)*s_ampa[t-1]) + n[:Ne]
        x[t] = x[t-1] + dt*((-1/tau_nmda_rise)*x[t-1]) + n[:Ne]
        s_nmda[t] = s_nmda[t-1] + dt*((-1/tau_nmda_decay)*s_nmda[t-1] + 500*x[t-1]*(1-s_nmda[t-1]))
        s_gaba[t] = s_gaba[t-1] + dt*((-1/tau_gaba)*s_gaba[t-1]) + n[Ne:]

        # incoming synaptic currents
        I_ext_ampa[t] = g_ext_ampa * (v[t] - Ve) * s_ext_ampa[t]
        I_ampa[t] = g_ampa * (v[t] - Ve) * (J[:,:Ne] @ s_ampa[t]) # matrix multiplication = excitatory to all other neurons
        I_nmda[t] = ((g_nmda * (v[t] - Ve)) / (1 + (np.exp(-0.062 * v[t]) / 3.57))) * (J[:,:Ne] @ s_nmda[t])
        I_gaba[t] = g_gaba * (v[t] - Vi) * (J[:,Ne:] @ s_gaba[t])
        I_syn[t] = I_ext_ampa[t] + I_ampa[t] + I_nmda[t] + I_gaba[t]

        v[t,spkind] = v_r
    
    gating_vars = {
        'ext_ampa': s_ext_ampa,
        'ampa': s_ampa,
        'nmda': s_nmda,
        'gaba': s_gaba
    }

    syn_currents = {
        'total': I_syn,
        'ext_ampa': I_ext_ampa,
        'ampa': I_ampa,
        'nmda': I_nmda,
        'gaba': I_gaba
    }

    return v, np.array(spktimes), spktrains, syn_currents, gating_vars

def gen_sensory_stim(mu_0=20, sigma=4.0, dt=None, rho=None, coh=None, f=0.15, N=2000, sim_len=4, stim_len=0.5, t_start=1):
    """
    Generate stimulus rates across time for Ns seconds. Rates are resampled every 50 ms.
    """
    if len(np.shape(rho)) > 0:
        rho_A,rho_B = rho
    else:
        rho_A = mu_0 / 100
        rho_B = mu_0 / 100
    mu_A = mu_0 + rho_A*coh
    mu_B = mu_0 - rho_B*coh

    print(f"A mean rate: {mu_A} Hz")
    print(f"B mean rate: {mu_B} Hz")
    print(f"Stimulation length: {stim_len} s")

    fNe = int(f * N * 0.8) # number of selective neurons

    Nt = int(sim_len / dt) # time points

    switch_idx = int(0.05 / dt) # every 50 ms
    num_switches = int((stim_len / dt) / switch_idx)

    stim_A_rate_list = np.random.normal(loc=mu_A, scale=sigma, size=num_switches)
    stim_B_rate_list = np.random.normal(loc=mu_B, scale=sigma, size=num_switches)
    
    stim_rates = np.zeros((Nt, N))
    stim_rates_A = np.zeros(Nt)
    stim_rates_B = np.zeros(Nt)

    stim_spikes = np.zeros((Nt, N))
    fNe2 = int(2*fNe)

    for i in range(num_switches):
        swi = int(i*switch_idx + (t_start / dt))
        
        stim_rates_A[swi:(swi+switch_idx)] = stim_A_rate_list[i]
        stim_rates_B[swi:(swi+switch_idx)] = stim_B_rate_list[i]

        stim_spikes[swi:(swi+switch_idx),:fNe] = (np.random.rand(switch_idx,fNe) < (stim_A_rate_list[i] * dt))*1
        stim_spikes[swi:(swi+switch_idx),fNe:fNe2] = (np.random.rand(switch_idx,fNe) < (stim_B_rate_list[i] * dt))*1

    stim_rates[:,:fNe] = stim_rates_A[:,np.newaxis].repeat([fNe], axis=1)
    stim_rates[:,fNe:fNe2] = stim_rates_B[:,np.newaxis].repeat([fNe], axis=1)

    return stim_rates, stim_spikes

def create_spike_train(spktimes, neuron=0, dt=.01, tstop=100):
    
    '''
    create a spike train from a list of spike times and neuron indices
    spktimes: Nspikes x 2, first column is times and second is neurons
    dt and tstop should match the simulation that created spktimes
    '''
    
    spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
    
    Nt = int(tstop/dt)+1
    spktrain = np.zeros((Nt,))
    
    spk_indices = spktimes_tmp / dt
    spk_indices = spk_indices.astype('int')

    spktrain[spk_indices] = 1/dt
    
    return spktrain


if __name__=='__main__':
    pass