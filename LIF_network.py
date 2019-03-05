
from brian2 import *
from matplotlib import *
import numpy as np
import random

##############
# Parameters #
##############

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
g_l_GrC   = 0.8   * nS
g_l_GoC   = 3.    * nS

# Reversal potential (leak, excitatory, inhibitory)
E_l_GrC   = -75   * mV
E_e_GrC   =   0   * mV
E_i_GrC   = -75   * mV

E_l_GoC   = -60   * mV
E_e_GoC   = -30   * mV
E_i_GoC   = -60   * mV

# Membrane capacitance
C_m_GrC   =  3.1  * pF
C_m_GoC   = 60.   * pF

# TODO Use rise and decay times and add dyanmics to neuron Equations
tau_e_decay_GrC = 2.9 * ms
tau_e_decay_GoC = 1.6 * ms

tau_i_decay_GrC = 8.0 * ms

# Absolute refractory period
tau_r = 2 * ms

# Spiking threshold
V_th_GrC   = -55 * mV
V_th_GoC   = -45 * mV

# Resting potential
V_r_GrC    = -75 * mV
V_r_GoC    = -60 * mV

# TODO: initialize synaptic weights as distribution to pull from
w_e = 0.34 * nS
w_i = 0.23 * nS

# Golgi cell tonic / stochastic current
g_t_GoC = 0.1  * nS
sigma_n = 0.12 * nS
tau_n   = 1000 * ms
#############
# Equations #
#############

GrC_eqs = '''
dv/dt   = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
'''

GoC_eqs = '''
dv/dt   = ((g_l+g_t)*(E_l-v) + (g_e+g_n)*(E_e-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_n/dt = (-g_n + sigma_n * sqrt(tau_n) * xi)/tau_n : siemens
'''
#####################
# Mossy fiber input #
#####################

stim_times = [0, 10, 20]            # Mossy fiber stimulation times
nstim      = len(stim_times)
mf_indices = np.arange(nmf)
n_active   = round(nmf/15)           # Fraction of mfs active at each stimulation

# Randomly select a subset of mossy fibers to spike at each stimulation
# If you want a different subset of mossy fibers at each stimulation, move
# the declaration of [active_indices] into the loop over stim_times
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
                  refractory = 'tau_r',
                  method     = 'euler')
GrC.v   = V_r_GrC

GoC = NeuronGroup(nGoC,
                  Equations(GoC_eqs,
                            g_l = g_l_GoC,
                            g_t = g_t_GoC,
                            E_l = E_l_GoC,
                            E_e = E_e_GoC,
                            E_i = E_i_GoC,
                            C_m = C_m_GoC,
                            tau_e = tau_e_decay_GoC),
                  threshold  = 'v > V_th_GoC',
                  reset      = 'v = V_r_GoC',
                  refractory = 'tau_r',
                  method     = 'euler')
GoC.v = V_r_GoC

###################
# Connect neurons #
###################

# Mossy fiber onto GrCs
GrC_M = Synapses(mf_input,GrC,
                 on_pre = 'g_e += w_e')
GrC_M.connect(p=float(conv_GrC_M/nmf))

# Mossy fiber onto GoCs
GoC_M = Synapses(mf_input,GoC,
                 on_pre = 'g_e += w_e')
GoC_M.connect(p=float(conv_GoC_M/nmf))

# GrC onto GoCs
GoC_GrC = Synapses(GrC,GoC,
                   on_pre = 'g_e += w_e')
GoC_GrC.connect(p=float(conv_GoC_GrC/nGrC))

# GoC onto GrC (inhibitory)
GrC_GoC = Synapses(GoC,GrC,
                   on_pre = 'g_i += w_i',
                   delay  = tau_r)
GrC_GoC.connect(p=float(conv_GrC_GoC/nGoC))

##############
# Simulation #
##############

# Monitor GrC output
spikes = SpikeMonitor(GrC)
state  = StateMonitor(GrC, 'v', record = GrC_M[active_indices[0],:])

runtime = 30
run(runtime * ms)

subplot(121);
plot(state.t/ms, np.transpose(state.v)/mV)


subplot(122);
plot(spikes.t/ms,spikes.i,'.k')
xlim(0,runtime); ylim(0,nGrC)
xlabel('Time (ms)'); ylabel('GrC index')

show()
