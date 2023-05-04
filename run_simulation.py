from brian2 import *
from matplotlib import *
from sympy import *
import numpy as np
import pandas as pd
import scipy.sparse
import random
import os

'''
Environment Variables
---------------------
CHAT_NUM_MF : int, default = 315
    number of mossy fiber inputs
CHAT_NUM_GRC : int, default = 4096
    number of granule cells
CHAT_NUM_GOC : int, default = 27
    number of golgi cells
CHAT_NUM_TRIAL : int, default = 100
    number of trials for the simulation
CHAT_RANDOM_WEIGHTS : boolean, default = True
    use random weights for synaptic connections
CHAT_SEED_TRIAL : int, default = 451
    seed value for trial runs
CHAT_SEED_WEIGHT_MEANS : int, default = 35
    seed value for generating the random synaptic weight 

Use:
$ export CHAT_NUM_TRIAL=50
$ export CHAT_NUM_TRIAL=100 CHAT_NUM_MF=315 CHAT_NUM_GRC=4096 CHAT_NUM_GOC=27 # 100um^3 (published: biologically relevant)
'''

def main():
    Conditions = ['Control','GABAzine','ACh']
    for Condition in Conditions:
        print('Running {} condition'.format(Condition))
        trial_sim(Condition)
    print('done')

def trial_sim(Condition):

    @check_units(mu=1,sigma=1,result=1)
    def gamma_parms(mu,sigma):
        a,theta = symbols('a theta')
        eqns = [Eq(a*theta,mu), Eq(a*theta**2,sigma**2)]
        vars = [a,theta]
        sols = nonlinsolve(eqns,vars)
        sol_a,sol_theta = next(iter(sols))
        shape = float(sol_a)
        scale = float(sol_theta)
        return shape, scale

    seed(int(os.getenv("CHAT_SEED_TRIAL", 451)))
    ##############
    # Parameters #
    ##############

    # Trial parameters
    runtime = 2000

    # Synaptic weight parameters
    random_weights = bool(os.getenv("CHAT_RANDOM_WEIGHTS", True))
    # weights_gamma_shape,weights_gamma_scale = gamma_parms(0.15,0.2)
    weights_gamma_shape,weights_gamma_scale = gamma_parms(0.1,0.1)

    # Cell numbers to reproduce functionally relevant 100um^3 cube of granular layer
    nmf  = int(os.getenv("CHAT_NUM_MF", 315))   # Mossy fibers
    nGrC = int(os.getenv("CHAT_NUM_GRC", 4096)) # Granule cells
    nGoC = int(os.getenv("CHAT_NUM_GOC", 27))   # Golgi cells

    # Convergence ratios of connections (to determine connection probabilities)
    conv_GrC_M   =   4          #  mf -> GrC synapses
    conv_GoC_M   =  10          #  mf -> GoC synapses
    conv_GoC_GrC = 100          # GrC -> GoC synapses
    conv_GrC_GoC =   4          # Goc -> GrC synapses

    # Leak conductance
    g_l_GrC   = 0.2   * nS
    g_t_GrC   = 1.0   * nS
    g_l_GoC   = 3.    * nS

    # Reversal potential (leak, excitatory, inhibitory)
    E_l_GrC   = -75   * mV
    E_e_GrC   =   0   * mV
    E_i_GrC   = -75   * mV

    E_l_GoC   = -51   * mV
    E_e_GoC   =   0   * mV
    E_i_GoC   = -75   * mV

    # Membrane capacitance
    C_m_GrC   =  3.1  * pF
    C_m_GoC   = 60.   * pF

    # Decay constants
    tau_e_decay_GrC = 12.0 * ms
    tau_e_decay_GoC = 12.0 * ms

    tau_i_decay_GrC = 20.0 * ms

    # Absolute refractory period
    tau_r_GrC = 2 * ms
    tau_r_GoC = 10* ms

    # Spiking threshold
    V_th_GrC   = -55 * mV
    V_th_GoC   = -50 * mV

    # Resting potential
    V_r_GrC    = -75 * mV
    V_r_GoC    = -55 * mV

    # Golgi cell reset potential
    V_reset_GoC = -55 * mV #-60

    # Synaptic weights
    fixed_w_e_GrC = 0.25 * nS #0.65 * nS
    w_i_GrC = 0.17 * nS

    fixed_w_e_GoC_M = 0.65 * nS #0.35  * nS
    w_e_GoC_GrC = 0.0 * nS

    # Stochastic fluctuating excitatory current
    sigma_n_GoC = 0.1 * nS
    sigma_n_GrC = 0.05 * nS

    tau_n   = 20 * ms

    #############
    # Equations #
    #############
    GrC_eqs = '''
    dv/dt   = (g_l*(E_l-v) + (g_e+g_n)*(E_e-v) + (g_i+g_t)*(E_i-v))/C_m : volt (unless refractory)
    dg_n/dt = (-g_n + sigma_n_GrC * sqrt(tau_n) * xi)/tau_n : siemens
    dg_e/dt = -g_e/tau_e : siemens
    dg_i/dt = -g_i/tau_i : siemens
    g_e_tot_GrC : siemens
    g_i_tot_GrC : siemens
    g_t = reduce_tonic * g_t_GrC : siemens
    reduce_tonic: 1
    '''

    GoC_eqs = '''
    dv/dt   = (g_l*(E_l-v) + (g_e+g_n)*(E_e-v))/C_m : volt (unless refractory)
    dg_e/dt = -g_e/tau_e : siemens
    dg_n/dt = (-g_n + sigma_n_GoC * sqrt(tau_n) * xi)/tau_n : siemens
    g_e_tot_GoC : siemens
    '''

    #####################
    # Mossy fiber input #
    #####################

    stim_times = [1010, 1020, 1030]            # Mossy fiber stimulation times
    nstim      = len(stim_times)
    mf_indices = np.arange(nmf)
    n_active   = round(nmf/10)           # Fraction of mfs active at each stimulation

    # Randomly select a subset of mossy fibers to spike at each stimulation
    # If you want a different subset of mossy fibers at each stimulation, move
    # the declaration of [active_indices] into the loop over stim_times

    random.seed(a=int(os.getenv("CHAT_SEED_TRIAL", 451)))
    active_indices = [mf_indices[i] for i in sorted(random.sample(range(len(mf_indices)),n_active))]
    indices    = []
    times      = []
    for j in range(nstim):
        indices.extend(active_indices)
        times.extend([stim_times[j]]*len(active_indices))
    times    = times * ms
    mf_input = SpikeGeneratorGroup(nmf, indices, times)


    ######################
    # Neuron populations #
    ######################
    if Condition == 'GABAzine':
        GrC = NeuronGroup(nGrC,
                          Equations(GrC_eqs,
                                    g_l   = g_l_GrC,
                                    E_l   = E_l_GrC,
                                    E_e   = E_e_GrC,
                                    E_i   = E_i_GrC,
                                    C_m   = C_m_GrC,
                                    tau_e = tau_e_decay_GrC,
                                    tau_i = tau_i_decay_GrC),
                          threshold  = 'v > V_th_GrC',
                          reset      = 'v = V_r_GrC',
                          refractory = 'tau_r_GrC',
                          method     = 'euler')
        GrC.v   = V_r_GrC
        GrC.reduce_tonic[:] = 0
    elif Condition == 'Control':
        GrC = NeuronGroup(nGrC,
                          Equations(GrC_eqs,
                                    g_l   = g_l_GrC,
                                    E_l   = E_l_GrC,
                                    E_e   = E_e_GrC,
                                    E_i   = E_i_GrC,
                                    C_m   = C_m_GrC,
                                    tau_e = tau_e_decay_GrC,
                                    tau_i = tau_i_decay_GrC),
                          threshold  = 'v > V_th_GrC',
                          reset      = 'v = V_r_GrC',
                          refractory = 'tau_r_GrC',
                          method     = 'euler')
        GrC.v   = V_r_GrC
        GrC.reduce_tonic[:] = 1
    elif Condition == 'ACh':
        GrC = NeuronGroup(nGrC,
                          Equations(GrC_eqs,
                                    g_l   = g_l_GrC,
                                    E_l   = E_l_GrC,
                                    E_e   = E_e_GrC,
                                    E_i   = E_i_GrC,
                                    C_m   = C_m_GrC,
                                    tau_e = tau_e_decay_GrC,
                                    tau_i = tau_i_decay_GrC),
                          threshold  = 'v > V_th_GrC',
                          reset      = 'v = V_r_GrC',
                          refractory = 'tau_r_GrC',
                          method     = 'euler')
        GrC.v   = V_r_GrC
        # Tonic reduction mean = 0.4 (60% reduction) variance = 0.07
        reduction_shape,reduction_scale = gamma_parms(0.4,0.07)
        GrC.reduce_tonic[:] = np.random.RandomState(seed=1).gamma(reduction_shape,reduction_scale,nGrC)
    else:
        print('ERROR: Unknown experimental condition')

    if Condition in ('Control','GABAzine'):
        GoC = NeuronGroup(nGoC,
                          Equations(GoC_eqs,
                                    g_l = g_l_GoC,
                                    E_l = E_l_GoC,
                                    E_e = E_e_GoC,
                                    E_i = E_i_GoC,
                                    C_m = C_m_GoC,
                                    tau_e = tau_e_decay_GoC),
                          threshold  = 'v > V_th_GoC',
                          reset      = 'v = V_r_GoC',
                          refractory = 'tau_r_GoC',
                          method     = 'euler')
    else:
        GoC = NeuronGroup(nGoC,
                          Equations(GoC_eqs,
                                    g_l = g_l_GoC,
                                    E_l = -55*mV,
                                    E_e = E_e_GoC,
                                    E_i = E_i_GoC,
                                    C_m = C_m_GoC,
                                    tau_e = tau_e_decay_GoC),
                          threshold  = 'v > V_th_GoC',
                          reset      = 'v = V_r_GoC',
                          refractory = 'tau_r_GoC',
                          method     = 'euler')
    GoC.v = V_r_GoC

    ###################
    # Connect neurons #
    ###################
    if Condition in ('Control', 'GABAzine'):
        # Mossy fiber onto GrCs
        GrC_M = Synapses(mf_input,GrC,
                         model  = '''w_e_GrC : siemens
                                     g_e_tot_GrC_post = g_e : siemens (summed)''',
                         on_pre = 'g_e += w_e_GrC')
        GrC_M.connect(p = conv_GrC_M/nmf)

        # Mossy fiber onto GoCs
        GoC_M = Synapses(mf_input,GoC,
                         model  = '''w_e_GoC_M : siemens
                                     g_e_tot_GoC_post = g_e : siemens (summed)''',
                         on_pre = 'g_e += w_e_GoC_M')
        GoC_M.connect(p = conv_GoC_M/nmf)
    elif Condition == 'ACh':
        # Mossy fiber onto GrCs
        GrC_M = Synapses(mf_input,GrC,
                         model  = '''w_e_GrC : siemens
                                     g_e_tot_GrC_post = g_e : siemens (summed)''',
                         on_pre = 'g_e += 0.4521*w_e_GrC')
        GrC_M.connect(p = conv_GrC_M/nmf)

        # Mossy fiber onto GoCs
        GoC_M = Synapses(mf_input,GoC,
                         model  = '''w_e_GoC_M : siemens
                                     g_e_tot_GoC_post = g_e: siemens (summed)''',
                         on_pre = 'g_e += 0.4578*w_e_GoC_M')
        GoC_M.connect(p = conv_GoC_M/nmf)

    # GoC onto GrC (inhibitory)
    if Condition == 'GABAzine':
        GrC_GoC = Synapses(GoC,GrC,
                           on_pre = 'g_i += 0 * nS',
                           delay  = tau_r_GrC)
    elif Condition in ('Control','ACh'):
        GrC_GoC = Synapses(GoC,GrC,
                           model  = 'g_i_tot_GrC_post = g_i : siemens (summed)',
                           on_pre = 'g_i += w_i_GrC',
                           delay  = tau_r_GrC)
    # seed(4321)
    GrC_GoC.connect(p = conv_GrC_GoC/nGoC)

    connections = pd.DataFrame({'MF': GrC_M.i[active_indices,:],'GrC': GrC_M.j[active_indices,:]})
    ffi_connections = pd.DataFrame({'GoC':GrC_GoC.i[:], 'GrC':GrC_GoC.j[:]})
    ################################
    # Randomize excitatory weights #
    ################################
    if random_weights:
        active_mf_GrC_weights = dict()
        active_mf_GoC_weights = dict()
        print('Randomizing synaptic weights')

        @network_operation(dt=runtime*ms, when='start')
        def update_input():
            weight_means = np.random.RandomState(seed=int(os.getenv("CHAT_SEED_WEIGHT_MEANS", 35))).gamma(weights_gamma_shape,weights_gamma_scale,n_active)
            # weight_sigma = 0.2
            tempidx = 0
            for i in active_indices:
                mu = weight_means[tempidx]
                # sigma = weight_sigma
                sigma = mu
                shape,scale = gamma_parms(mu,sigma)
                weight = np.random.RandomState(seed=trial_seed).gamma(shape,scale)*nS
                GrC_M.w_e_GrC[  i,:] = weight
                GoC_M.w_e_GoC_M[i,:] = weight
            # Fill in weight dictionary by mossy fiber number, add new mf key if non-existent
                if i in active_mf_GrC_weights.keys():
                    active_mf_GrC_weights[i].append(weight/nS)
                    active_mf_GoC_weights[i].append(weight/nS)
                else:
                    active_mf_GrC_weights[i] = [weight/nS]
                    active_mf_GoC_weights[i] = [weight/nS]
                tempidx += 1
    else:
        print('Using fixed synaptic weights')
        GrC_M.w_e_GrC[active_indices,:]   = fixed_w_e_GrC
        GoC_M.w_e_GoC_M[active_indices,:] = fixed_w_e_GoC_M



    ##############
    # Simulation #
    ##############

    # Set up granule cell spike monitor to collect spike trains
    M = SpikeMonitor(GrC)
    state_GrC  = StateMonitor(GrC, ['g_e_tot_GrC','g_i_tot_GrC'], dt=1*ms, record = True)
    # Store network state
    store(Condition)
    # Initialize matrix to store spike spike trains and conductance traces
    spikes = []
    cond_e_mats = []
    cond_i_mats = []
    # Number of trials to simulate
    ntrial = int(os.getenv("CHAT_NUM_TRIAL", 100))
    seeds = np.random.RandomState(seed=43).randint(1,1000,ntrial)
    for trial in range(ntrial):
        print('Trial {}'.format((trial+1)))
        # Restore network state
        restore(Condition)
        trial_seed = seeds[trial]
        # Run simulation and print progress report to console
        run(runtime * ms, report='stdout', report_period=1*second,
            profile=True)
        # Save spike times for each simulation
        spike_trains = M.spike_trains()
        spikes.append(spike_trains)
        # Save excitatory and inhibitory conductance traces for each simulation
        cond_e = state_GrC.g_e_tot_GrC/nS
        cond_i = state_GrC.g_i_tot_GrC/nS
        cond_e_mats.append(cond_e)
        cond_i_mats.append(cond_i)

    ############
    # Analysis #
    ############
    dt     = 1e-3
    ncol = int(runtime*1e-3/dt)
    mats   = []
    tuples = []
    for trial_idx  in range(ntrial):
        nrow    = len(spikes[trial_idx])
        arrays  = [np.full(nrow,trial_idx),range(nrow)]
        tuples += list(zip(*arrays))

        row  = np.concatenate([[key]*len(val) for key,val in spikes[trial_idx].items()])
        data = np.concatenate([val for key,val in spikes[trial_idx].items()])
        col  = data/dt
        trial_mat  = scipy.sparse.csr_matrix((data,(row.astype(int), col.astype(int))),shape = (nrow,ncol))
        mats.append(trial_mat)

    multi_index = pd.MultiIndex.from_tuples(tuples,names=['trial','neuron'])

    mat = scipy.sparse.vstack(mats)
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=multi_index)

    mat_cond_e = np.vstack(cond_e_mats)
    df_cond_e = pd.DataFrame(mat_cond_e,index=multi_index)

    mat_cond_i = np.vstack(cond_i_mats)
    df_cond_i = pd.DataFrame(mat_cond_i,index=multi_index)

    spike_prob = np.zeros((nGrC,ncol))
    mean_cond_e = np.zeros((nGrC,mat_cond_e.shape[1]))
    mean_cond_i = np.zeros((nGrC,mat_cond_i.shape[1]))
    for neuron in range(nGrC):
        neuron_count = neuron+1
        if neuron_count < nGrC:
            print('{}: Compiling spike trains for neuron {} of {}'.format(Condition,neuron_count,nGrC), end='\r')
        else:
            print('{}: Compiling spike trains for neuron {} of {}'.format(Condition,neuron_count,nGrC), end='\n')
        st = df.loc[(slice(None),neuron),:].to_numpy()
        x = np.nan_to_num(st).astype(bool)
        p = np.mean(x,axis=0)
        spike_prob[neuron,:] = p

        this_cond_e = df_cond_e.loc[(slice(None),neuron),:].to_numpy()
        mean_this_cond_e = np.mean(this_cond_e,axis=0)
        mean_cond_e[neuron,:] = mean_this_cond_e

        this_cond_i = df_cond_i.loc[(slice(None),neuron),:].to_numpy()
        mean_this_cond_i = np.mean(this_cond_i,axis=0)
        mean_cond_i[neuron,:] = mean_this_cond_i

    mean_cond_e = mean_cond_e[np.unique(GrC_M.j[active_indices,:]),:]
    mean_cond_i = mean_cond_i[np.unique(GrC_M.j[active_indices,:]),:]

    if Condition == 'Control':
        if random_weights:
            np.save('randomized_controlSpikeProb',spike_prob)
            np.save('controlConnections',connections)
            np.save('controlWeights',active_mf_GrC_weights)
            np.save('controlCondE',mean_cond_e)
            np.save('controlCondI',mean_cond_i)
        else:
            np.save('controlSpikeProb',spike_prob)
    elif Condition == 'GABAzine':
        if random_weights:
            np.save('randomized_gabaSpikeProb',spike_prob)
            np.save('gabaConnections',connections)
            np.save('gabaWeights', active_mf_GrC_weights)
            np.save('gabaCondE',mean_cond_e)
            np.save('gabaCondI',mean_cond_i)
        else:
            np.save('gabaSpikeProb',spike_prob)
    elif Condition == 'ACh':
        if random_weights:
            np.save('randomized_achSpikeProb',spike_prob)
            np.save('achConnections',connections)
            np.save('achWeights', active_mf_GrC_weights)
            reduce_tonic = GrC.reduce_tonic[np.unique(GrC_M.j[active_indices,:])]
            np.save('tonicReduction',reduce_tonic)
            np.save('achCondE',mean_cond_e)
            np.save('achCondI',mean_cond_i)
        else:
            np.save('achSpikeProb',spike_prob)
    else:
        print('ERROR: Unknown experimental condition')

if __name__ == '__main__':
    main()
