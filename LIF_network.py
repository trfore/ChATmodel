from brian2 import *
from matplotlib import *
import numpy as np
import random
from fractions import Fraction
seed(451)
##############
# Parameters #
##############
Condition = 'Control' # Control, GABAzine, ACh

# Trial parameters
runtime = 2000

# Synaptic weight parameters
random_weights    = True
GrC_M_gamma_shape = 1.5625
GrC_M_gamma_scale = 0.16
GoC_M_gamma_shape = 1.5625
GoC_M_gamma_scale = 0.16

# Cell numbers to reproduce functionally relevant 100um^3 cube of granular layer
nmf  =  315                 # Mossy fibers
nGrC = 4096                 # Granule cells
nGoC =   27                 # Golgi cells

# Convergence ratios of connections (to determine connection probabilities)
conv_GrC_M   =   4          #  mf -> GrC synapses
conv_GoC_M   =  10          #  mf -> GoC synapses
conv_GoC_GrC = 100          # GrC -> GoC synapses
conv_GrC_GoC =   4          # Goc -> GrC synapses

# Leak conductance
g_l_GrC   = 0.4   * nS
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
    GrC.reduce_tonic[:] = np.random.gamma(32.6531,0.0123,nGrC)
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
                       model  = 'g_i_tot_GrC_post = g_i : siemens (summed)',
                       on_pre = 'g_i += w_i_GrC',
                       delay  = tau_r_GrC)
GrC_GoC.connect(p = conv_GrC_GoC/nGoC)

################################
# Randomize excitatory weights #
################################
if random_weights:
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

# Monitor output
spikes_GrC = SpikeMonitor(GrC)
state_GrC  = StateMonitor(GrC, ['v','g_e_tot_GrC','g_i_tot_GrC'], record = True)
LFP        = PopulationRateMonitor(GrC)
spikes_GoC = SpikeMonitor(GoC)
state_GoC  = StateMonitor(GoC, ['v','g_e_tot_GoC'], record = True)


# Run simulation
run(runtime * ms)

spikes = spikes_GrC.spike_trains()
row = np.concatenate([[key]*len(val) for key,val in spikes.items()])
to_plot = np.unique(row).astype(int)
num_active = len(to_plot)
fraction_active = Fraction(num_active,nGrC)
print('Active fraction: ', num_active,'/',nGrC)

unique_GrC,input_counts = np.unique(GrC_M.j[active_indices,:],return_counts=True)
unique_counts,num_cells = np.unique(input_counts,return_counts=True)
print(dict(zip(unique_counts,num_cells)))
# Plots
fig, ax = plt.subplots(2, 3, figsize=(9, 9))
for g in to_plot:
    ax[0,0].plot(state_GrC.t/ms, state_GrC.v[g]/mV)
ax[0,0].set_title('GrC Vm (population)')
ax[0,0].set_xlim(stim_times[0]-10,stim_times[-1]+30)
ax[0,0].set_xlabel('time (ms)')
ax[0,0].set_ylabel('Vm (mV)')

ax[0,1].plot(spikes_GrC.t/ms,spikes_GrC.i,'.k')
for stim in range(nstim):
    ax[0,1].axvline(stim_times[stim], ls = '-', color = 'r', lw = 2)
ax[0,1].set_xlim(stim_times[0]-10,stim_times[-1]+30)
ax[0,1].set_ylim(0,nGrC)
ax[0,1].set_title('GrC population activity')
ax[0,1].set_xlabel('time (ms)')
ax[0,1].set_ylabel('GrC index')

for g in to_plot:
    ax[0,2].plot(state_GrC.t/ms, state_GrC.g_e_tot_GrC[g]*(-70*mV-E_e_GrC)/pA)
    ax[0,2].plot(state_GrC.t/ms, state_GrC.g_i_tot_GrC[g]*(10*mV-E_i_GrC)/pA)
ax[0,2].set_title('GrC EPSC/IPSC')
ax[0,2].set_xlim(stim_times[0]-10,stim_times[-1]+30)
ax[0,2].set_xlabel('time (ms)')
ax[0,2].set_ylabel('I (pA)')


for g in range(nGoC):
    ax[1,0].plot(state_GoC.t/ms, state_GoC.v[g]/mV)
ax[1,0].set_title('GoC Vm (population)')
ax[1,0].set_xlabel('time (ms)')
ax[1,0].set_ylabel('Vm (mV)')
ax[1,0].set_xlim(stim_times[0]-10,stim_times[-1]+30)

ax[1,1].plot(spikes_GoC.t/ms,spikes_GoC.i,'.k')
ax[1,1].set_xlim(stim_times[0]-10,stim_times[-1]+30)
ax[1,1].set_ylim(0,nGoC)
ax[1,1].set_title('GoC population activity')
ax[1,1].set_xlabel('time (ms)')
ax[1,1].set_ylabel('GoC index')

for g in range(nGoC):
    ax[1,2].plot(state_GoC.t/ms, state_GoC.g_e_tot_GoC[g]*(-60*mV-E_e_GrC)/pA)
ax[1,2].set_title('GoC EPSC')
ax[1,2].set_xlabel('time(ms)')
ax[1,2].set_ylabel('I (pA)')
ax[1,2].set_xlim(stim_times[0]-10,stim_times[-1]+30)


plt.figure()
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(stim_times[0]-10,stim_times[-1]+30)
show(block=False)
