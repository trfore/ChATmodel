import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

random_weights = True

# Import matrices of individual cell spike probabilities saved for each
# experimental condition
if random_weights:
    control_all = np.load('randomized_controlSpikeProb.npy')
    gaba_all    = np.load('randomized_gabaSpikeProb.npy')
    ach_all     = np.load('randomized_achSpikeProb.npy')
else:
    control_all = np.load('controlSpikeProb.npy')
    gaba_all    = np.load('gabaSpikeProb.npy')
    ach_all     = np.load('achSpikeProb.npy')

# Get rid of entirely quiescent cells
indices_nonzero = ~np.all(control_all==0,axis=1)

control = control_all[indices_nonzero,1000:1060]
gaba = gaba_all[indices_nonzero,1000:1060]
ach = ach_all[indices_nonzero,1000:1060]

ncells = control.shape[0]
nt     = control.shape[1]

# Normalize to control condition (max-min)
controlmax = np.reshape(np.amax(control,axis=1),(ncells,1))

controlnorm = control/controlmax
gabanorm    = gaba/controlmax
achnorm     = ach/controlmax

# Average
max_ach = np.amax(achnorm,axis=1)
ach_increase = max_ach > 1

mean_control_up = np.mean(controlnorm[ach_increase,:],axis=0)
mean_gaba_up    = np.mean(gabanorm[ach_increase,:],axis=0)
mean_ach_up     = np.mean(achnorm[ach_increase,:],axis=0)

mean_control_down = np.mean(controlnorm[~ach_increase,:],axis=0)
mean_gaba_down    = np.mean(gabanorm[~ach_increase,:],axis=0)
mean_ach_down     = np.mean(achnorm[~ach_increase,:],axis=0)

stim_times = [1010, 1020, 1030]

fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].plot(np.arange(nt),mean_control_up,'k')
ax[0].plot(np.arange(nt),mean_gaba_up,'r')
ax[0].plot(np.arange(nt),mean_ach_up,'b')


ax[1].plot(np.arange(nt),mean_control_down,'k')
ax[1].plot(np.arange(nt),mean_gaba_down,'r')
ax[1].plot(np.arange(nt),mean_ach_down,'b')

x = np.amax(control,axis=1)
y = np.amax(ach, axis=1)
c = Counter(zip(x,y))
s = [2*c[(xx,yy)] for xx,yy in zip(x,y)]
ax[2].scatter(x,y,c='k',s=s)
ax[2].set_xlabel('Control peak spike probability')
ax[2].set_ylabel('Muscarine peak spike probability')
start,stop = ax[2].get_xlim()
x = np.linspace(start,stop)
y = x
ax[2].plot(x,y,'--k')
plt.show(block=False)
