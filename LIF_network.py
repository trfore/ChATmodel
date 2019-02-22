
from brian2 import *
from matplotlib import *
import numpy as np

# Cell numbers to reproduce functionally relevant 100um^3 cube of granular layer
nmf  = 315                  # Mossy fibers
nGrC = 4096                 # Granule cells
nGoC = 27                   # Golgi cells

# Convergence ratios of connections (to determine connection probabilities)
conv_GrC_M   = 4            #  mf -> GrC synapses
conv_GoC_M   = 50           #  mf -> GoC synapses
conv_GoC_GrC = 100          # GrC -> GoC synapses
conv_GrC_GoC = 4            # Goc -> GrC synapses

g_l   = 0.789 * nS          # Leak conductance

E_l   = -75   * mV          # Leak reversal potential
E_e   =   0   * mV          # Excitatory reversal potential
E_i   = -75   * mV          # Inhibitory reversal potential

C_m   = 3.1   * pF          # Membrane capacitance

tau_e = 2.5   * ms          # excitatory current decay
tau_i = 6.2   * ms          # inhibitory current decay
tau_r = 2.5   * ms          # refractory period

V_th   = -55   * mV         # Spiking threshold
V_r   = -75   * mV          # Resting potential

w_e = 0.05 * nS             # Excitatory conductance
w_i = 1 * nS                # Inhibitory conductance

#############
# Equations #
#############

synapse_eqs = '''
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
'''

neuron_eqs = Equations('''
dv/dt = (g_l*(E_l-v) + g_e*(E_e - v) + g_i*(E_i-v))/C_m : volt (unless refractory)
''' + synapse_eqs)

#####################
# Mossy fiber input #
#####################

input_frequencies = np.full(nmf, 30)
mf_input = PoissonGroup(nmf, input_frequencies*Hz)

######################
# Neuron populations #
######################

GrC = NeuronGroup(nGrC,
                  neuron_eqs,
                  threshold  = 'v > V_th',
                  reset      = 'v = V_r',
                  refractory = 'tau_r',
                  method     = 'euler')
GrC.v   = V_r
GrC.g_e = 3.25 * nS
GrC.g_i = 2.0 * nS

#GoC = NeuronGroup(nGoC,
#                  neuron_eqs,
#                  threshold  = 'v > V_th',
#                  reset      = 'v = V_r',
#                  refractory = 'tau_r',
#                  method     = 'euler')

###################
# Connect neurons #
###################

# Mossy fiber synapses onto GrCs
GrC_M = Synapses(mf_input,GrC,
                 on_pre = 'g_e += w_e')
GrC_M.connect(p=float(conv_GrC_M/nmf))


## Mossy fiber synapses onto GoCs
#GoC_M = Synapses(mf_input,GoC,
#                 on_pre = 'g_e += w_e')
#GoC_M.connect(p=float(conv_GoC_M/nmf))

## GrC synapses onto GoCs
#GoC_GrC = Synapses(GrC,GoC,
#                   on_pre = 'g_e += w_e')
#GoC_GrC.connect(p=float(conv_GoC_GrC/nGrC))

## GoC synapses onto GrC (inhibitory)
#GrC_GoC = Synapses(GoC,GrC,
#                   on_pre = 'g_i += w_i')
#GrC_GoC.connect(p=float(conv_GrC_GoC/nGoC))

##############
# Simulation #
##############

# Monitor spike output of GrCs
spikes = SpikeMonitor(GrC)
state  = StateMonitor(GrC, 'v', record = True)
runtime = 30
run(runtime*ms)


subplot(121);
for i in range(0,20):
    plot(state.t/ms, state.v[i]/mV)
subplot(122);
plot(spikes.t/ms,spikes.i,'.k')
xlim(0,runtime); ylim(0,nGrC)
xlabel('Time (ms)'); ylabel('GrC index')
show()
