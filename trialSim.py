from brian2 import *
from matplotlib import *
import numpy as np
import random


##############
# Parameters #
##############

# RNG seed
seed(451)

# Experimental condition to simulate ('Control', 'GABAzine', 'ACh')
Condition = 'ACh'

# Trial parameters
runtime = 2000
ntrial = 30

# Synaptic weight parameters
random_MF_weight_gamma = True # Logical, if True a random MF w_e weight will be
                              # determined every trial. IF false, the values
                              #fixed_GrC_mf_w_e and fixed_GoC_mf_w_e will be
                              #used
GrC_M_gamma_shape = 42.25
GrC_M_gamma_scale = 0.0154
GoC_M_gamma_shape = 12.25
GoC_M_gamma_scale = 0.0286

# Cell numbers to reproduce functionally relevant 100um^3 cube of granular layer
nmf  =  315                 # Mossy fibers
nGrC = 4096                 # Granule cells
nGoC =   27                 # Golgi cells

# Convergence ratios of connections (to determine connection probabilities)
conv_GrC_M   =   4          #  mf -> GrC synapses
conv_GoC_M   =  50          #  mf -> GoC synapses
conv_GoC_GrC = 100          # GrC -> GoC synapses
conv_GrC_GoC =   4          # Goc -> GrC synapses

# Leak conductance
g_l_GrC   = 0.4   * nS
g_t_GrC   = 0.4   * nS

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

# TODO Use rise and decay times and add dyanmics to neuron Equations
tau_e_decay_GrC = 6.0 * ms
tau_e_decay_GoC = 3.0 * ms

tau_i_decay_GrC = 15.0 * ms

# Absolute refractory period
tau_r_GrC = 2 * ms
tau_r_GoC = 40* ms

# Spiking threshold
V_th_GrC   = -55 * mV
V_th_GoC   = -50 * mV

# Resting potential
V_r_GrC    = -75 * mV
V_r_GoC    = -55 * mV

# Golgi cell reset potential
V_reset_GoC = -60 * mV

# Synaptic weights
fixed_w_e_GrC = 0.65 * nS
w_i_GrC = 0.08 * nS

fixed_w_e_GoC_M = 0.35  * nS
w_e_GoC_GrC = 0.0 * nS

# Stochastic fluctuating excitatory current
sigma_n_GoC = 0.1 * nS
sigma_n_GrC = 0.03 * nS

tau_n   = 20 * ms

#############
# Equations #
#############

GrC_eqs = '''
dv/dt   = (g_l*(E_l-v) + (g_e+g_n)*(E_e-v) + (g_i+g_t)*(E_i-v))/C_m : volt (unless refractory)
dg_n/dt = (-g_n + sigma_n_GrC * sqrt(tau_n) * xi)/tau_n : siemens
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
'''

GoC_eqs = '''
dv/dt   = (g_l*(E_l-v) + (g_e+g_n)*(E_e-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_n/dt = (-g_n + sigma_n_GoC * sqrt(tau_n) * xi)/tau_n : siemens
'''

#####################
# Mossy fiber input #
#####################

stim_times = [1010, 1020, 1030]            # Mossy fiber stimulation times
nstim      = len(stim_times)
mf_indices = np.arange(nmf)
n_active   = round(nmf/20)           # Fraction of mfs active at each stimulation

# Randomly select a subset of mossy fibers to spike at each stimulation
# If you want a different subset of mossy fibers at each stimulation, move
# the declaration of [active_indices] into the loop over stim_times
random.seed(a=451)
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
                                g_t   = 0*nS,
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
elif Condition == 'Control':
    GrC = NeuronGroup(nGrC,
                    Equations(GrC_eqs,
                            g_l   = g_l_GrC,
                            g_t   = g_t_GrC,
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
elif Condition == 'ACh':
    GrC = NeuronGroup(nGrC,
                    Equations(GrC_eqs,
                        g_l   = g_l_GrC,
                        g_t   = 0.5 * g_t_GrC,
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
                      reset      = 'v = V_reset_GoC',
                      refractory = 'tau_r_GoC',
                      method     = 'euler')
    GoC.v = V_r_GoC
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
                      reset      = 'v = V_reset_GoC',
                      refractory = 'tau_r_GoC',
                      method     = 'euler')
GoC.v = V_r_GoC
###################
# Connect neurons #
###################
if Condition in ('Control', 'GABAzine'):
    # Mossy fiber onto GrCs
    GrC_M = Synapses(mf_input,GrC,
                     model  = 'w_e_GrC : siemens',
                     on_pre = 'g_e += w_e_GrC')
    GrC_M.connect(p = conv_GrC_M/nmf)

    # Mossy fiber onto GoCs
    GoC_M = Synapses(mf_input,GoC,
                     model  = 'w_e_GoC_M : siemens',
                     on_pre = 'g_e += w_e_GoC_M')
    GoC_M.connect(p = conv_GoC_M/nmf)

    # GrC onto GoCs
    GoC_GrC = Synapses(GrC,GoC,
                       on_pre = 'g_e += w_e_GoC_GrC')
    GoC_GrC.connect(p = conv_GoC_GrC/nGrC)
elif Condition == 'ACh':
    # Mossy fiber onto GrCs
    GrC_M = Synapses(mf_input,GrC,
                     model  = 'w_e_GrC : siemens',
                     on_pre = 'g_e += 0.4521*w_e_GrC')
    GrC_M.connect(p = conv_GrC_M/nmf)

    # Mossy fiber onto GoCs
    GoC_M = Synapses(mf_input,GoC,
                     model  = 'w_e_GoC_M : siemens',
                     on_pre = 'g_e += 0.4578*w_e_GoC_M')
    GoC_M.connect(p = conv_GoC_M/nmf)

    # GrC onto GoCs
    GoC_GrC = Synapses(GrC,GoC,
                       on_pre = 'g_e += 0.4578*w_e_GoC_GrC')
    GoC_GrC.connect(p = conv_GoC_GrC/nGrC)

    # GoC onto GrC (inhibitory)
    GrC_GoC = Synapses(GoC,GrC,
                       on_pre = 'g_i += w_i_GrC',
                       delay = tau_r_GrC)
    GrC_GoC.connect(p = conv_GrC_GoC/nGoC)

# GoC onto GrC (inhibitory)
if Condition == 'GABAzine':
    GrC_GoC = Synapses(GoC,GrC,
                       on_pre = 'g_i += 0 * nS',
                       delay  = tau_r_GrC)
elif Condition == 'Control':
    GrC_GoC = Synapses(GoC,GrC,
                       on_pre = 'g_i += w_i_GrC',
                       delay  = tau_r_GrC)

GrC_GoC.connect(p = conv_GrC_GoC/nGoC)

################################
# Randomize excitatory weights #
################################
if random_MF_weight_gamma is True:
    active_mf_GrC_weights = dict()
    active_mf_GoC_weights = dict()
    print('Randomizing synaptic weights')

    @network_operation(dt=runtime*ms, when='start')
    def update_input():
        for i in active_indices:
            temp_w_GrC_M = np.random.gamma(GrC_M_gamma_shape,
                                           GrC_M_gamma_scale) * nS
            temp_w_GoC_M = np.random.gamma(GoC_M_gamma_shape,
                                           GoC_M_gamma_scale) * nS
            GrC_M.w_e_GrC[  i,:] = temp_w_GrC_M
            GoC_M.w_e_GoC_M[i,:] = temp_w_GoC_M
        # Fill in weight dictionary by mossy fiber number, add new mf key if non-existent
            if i in active_mf_GrC_weights.keys():
                active_mf_GrC_weights[i].append(temp_w_GrC_M)
                active_mf_GoC_weights[i].append(temp_w_GoC_M)
            else:
                active_mf_GrC_weights[i] = [temp_w_GrC_M]
                active_mf_GoC_weights[i] = [temp_w_GoC_M]
else:
    print('Using fixed synaptic weights')
    GrC_M.w_e_GrC[active_indices,:]   = fixed_w_e_GrC
    GoC_M.w_e_GoC_M[active_indices,:] = fixed_w_e_GoC_M

##############
# Simulation #
##############
M = SpikeMonitor(GrC)
state_GrC  = StateMonitor(GrC, 'v', record = True)
store(Condition)

spikes = []
ntrial = 30
for trial in range(ntrial):
    print('Trial %d' % (trial+1))
    restore(Condition)

    # Run simulation
    runtime = 2000
    run(runtime * ms, report='stdout', report_period=1*second,
        profile=True)

    # Save spike times for each simulation
    spike_trains = M.spike_trains()
    spikes.append(spike_trains)

############
# Analysis #
############
import pandas as pd
import scipy.sparse
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

mat = scipy.sparse.vstack(mats)
multi_index = pd.MultiIndex.from_tuples(tuples,names=['trial','neuron'])
df = pd.SparseDataFrame(mat,index=multi_index)

spike_prob = np.zeros((nGrC,ncol))

for neuron in range(nGrC):
    neuron_count = neuron+1
    print('Compiling spike trains for neuron {}'.format(neuron_count), end='\r')
    st = df.loc[(slice(None),neuron),:].to_numpy()
    x = np.nan_to_num(st).astype(bool)
    p = np.sum(x,axis=0)/x.shape[0]
    spike_prob[neuron,:] = p

if Condition == 'Control':
    if random_MF_weight_gamma:
        np.save('randomized_controlSpikeProb',spike_prob)
    else:
        np.save('controlSpikeProb',spike_prob)
elif Condition == 'GABAzine':
    if random_MF_weight_gamma:
        np.save('randomized_gabaSpikeProb',spike_prob)
    else:
        np.save('gabaSpikeProb',spike_prob)
elif Condition == 'ACh':
    if random_MF_weight_gamma:
        np.save('randomized_achSpikeProb',spike_prob)
    else:
        np.save('achSpikeProb',spike_prob)
else:
    print('ERROR: Unknown experimental condition')

# Generate a dataframe with all of the Mossy Fiber to Granule cell Connections
connections = pd.DataFrame({'MF':GrC_M.i[:],'GrC':GrC_M.j[:]})

# Only look at MF-GrC pairs that are active during the stimulation
connections.loc[connections['MF'].isin(mf_indices)].drop_duplicates(['GrC'])
