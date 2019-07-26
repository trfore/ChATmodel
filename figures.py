import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KernelDensity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import rc

# fonttype setting required for CorelDraw compatibility
rc('pdf',fonttype=42)

##########################################################################
# Main program to run analysis and create figures.                       #
# Requires that you first run trialSim.py for each simulation condition. #
# Run by typing 'python figures.py'  in the terminal window.             #
# Figures will not display but will be saved to the folder               #
# this file is saved to.                                                 #
##########################################################################
def main():
    controlnorm, gabanorm, achnorm, max_ach, \
    control, ach, \
    unique, counts, \
    int_g_e, int_g_i, \
    reduce_tonic = run_analysis()

    plot_spike_prob(controlnorm, gabanorm, achnorm, max_ach, control, ach)
    plot_connections_summary(counts, max_ach, int_g_e, int_g_i, reduce_tonic)
    plot_conductance_scatter_heatmap(int_g_e, int_g_i, max_ach)
    plot_input_distributions(int_g_e, int_g_i,reduce_tonic, max_ach)
    plot_kde_inputs(int_g_e, int_g_i,reduce_tonic, max_ach)
    plot_inputs_summary(int_g_e, int_g_i, max_ach, reduce_tonic)

###########################################################
# Run analysis and return data required to create figures #
###########################################################
def run_analysis():
    # Load arrays listing all active MF->GrC pairs
    control_connections = np.load('controlConnections.npy')
    ach_connections     = np.load('achConnections.npy')
    # Check to make sure connections are the same for both conditions
    shared_connections = control_connections == ach_connections
    if np.all(shared_connections):
        connections = control_connections
    else:
        print('WARNING: Connections are not the same between conditions')

    # Calculate number of unique granule cells receiving active MF input
    GrC = connections[:,1]
    unique,counts = np.unique(GrC,return_counts=True)

    # Load spike probability arrays
    control_all = np.load('randomized_controlSpikeProb.npy')
    gaba_all    = np.load('randomized_gabaSpikeProb.npy')
    ach_all     = np.load('randomized_achSpikeProb.npy')

    nt = control_all.shape[1]

    # Only consider unique GrCs receiving active MF input
    control_unique = control_all[unique,:]
    gaba_unique    = gaba_all[unique,:]
    ach_unique     = ach_all[unique,:]

    # Get rid of entirely quiescent cells in control
    indices_nonzero = ~np.all(control_unique==0,axis=1)
    control_nonzero = control_unique[indices_nonzero,1000:1060]
    gaba_nonzero    = gaba_unique[indices_nonzero,1000:1060]
    ach_nonzero     = ach_unique[indices_nonzero,1000:1060]

    # Update cell counts
    unique = unique[indices_nonzero]
    counts = counts[indices_nonzero]

    # Now select cells for spiking in control that matches experiment
    max_stim_two   = np.max(control_nonzero[:,18:22],axis=1)
    max_stim_three = np.max(control_nonzero[:,28:32],axis=1)

    selection_criteria = (max_stim_two <= 0.6) & (max_stim_three <= max_stim_two)

    control = control_nonzero[selection_criteria,:]
    gaba    = gaba_nonzero[selection_criteria,:]
    ach     = ach_nonzero[selection_criteria,:]

    # Update cell counts again
    unique = unique[selection_criteria]
    counts = counts[selection_criteria]

    # Normalize to control condition (max-min)
    controlmax = np.max(control,axis=1)
    controlnorm = control / controlmax[:,None]
    gabanorm    = gaba / controlmax[:,None]
    achnorm     = ach / controlmax[:,None]

    # Maximum normalized spike probability in muscarine
    max_ach = np.max(achnorm,axis=1)


    # Load synaptic weight dictionaries
    control_weights     = np.load('controlWeights.npy')
    ach_weights         = np.load('achWeights.npy')
    x = control_weights.item()
    y = ach_weights.item()

    # Check to make sure weights are consistent for both conditions
    shared_weights = {k: x[k] for k in x if k in y and x[k]==y[k]}
    if len(shared_weights) < len(x):
        print('WARNING: Weights are not the same between conditions')
    else:
        weights = x

    # Load excitatory and inhibitory conductance traces
    g_e = np.load('controlCondE.npy')
    g_i = np.load('controlCondI.npy')
    if g_e.shape[1] != nt:
        print('WARNING: StateMonitor sampling does not match resolution of spiking data')
    else:
        start = 1000
        stop  = 1060
    # Make sure you only saved traces for unique GrCs receiving active MF input
    if g_e.shape[0] != control_unique.shape[0]:
        print('WARNING: Did not save excitatory conductances for only unique GrCs')
    else:
        g_e = g_e[indices_nonzero,start:stop]
        g_e = g_e[selection_criteria,:]

    if g_i.shape[0] != control_unique.shape[0]:
        print('WARNING: Did not save inhibitory conductances for only unique GrCs')
    else:
        g_i = g_i[indices_nonzero,start:stop]
        g_i = g_i[selection_criteria,:]

    #Integrate conductance traces
    nsamp = stop-start
    xval  = np.linspace(0,60,nsamp)
    int_g_e = np.trapz(g_e,xval,axis=1)
    int_g_i = np.trapz(g_i,xval,axis=1)

    # Load array with the fraction of tonic inhibition in muscarine
    reduce_tonic = np.load('tonicReduction.npy')
    if len(reduce_tonic) != control_unique.shape[0]:
        print('WARNING: Did not save tonic reduction for only unique GrCs')
    else:
        reduce_tonic = reduce_tonic[indices_nonzero]
        reduce_tonic = reduce_tonic[selection_criteria]

    return controlnorm, gabanorm, achnorm, max_ach, \
           control, ach, \
           unique, counts, \
           int_g_e, int_g_i, \
           reduce_tonic

########################################################
# Code to construct spike probability figure;          #
# Panel 1: Normalized spike probability plot for cells #
# with increasing probability in muscarine.            #
# Panel 2: Normalized spike probability plot for cells #
# with decreasing probaility in muscarine.             #
# Panel 3: Summary scatter plot of peak probability    #
# in muscarine vs. control                             #
########################################################
def plot_spike_prob(controlnorm, gabanorm, achnorm, max_ach, control, ach):
    ach_increase = max_ach > 1
    ach_decrease  = max_ach < 1
    n_increase = np.sum(ach_increase)
    n_decrease = np.sum(ach_decrease)

    mean_control_up = np.mean(controlnorm[ach_increase,:],axis=0)
    mean_gaba_up    = np.mean(gabanorm[ach_increase,:],axis=0)
    mean_ach_up     = np.mean(achnorm[ach_increase,:],axis=0)

    mean_control_down = np.mean(controlnorm[ach_decrease,:],axis=0)
    mean_gaba_down    = np.mean(gabanorm[ach_decrease,:],axis=0)
    mean_ach_down     = np.mean(achnorm[ach_decrease,:],axis=0)

    nt = controlnorm.shape[1]

    stim_times = [1010, 1020, 1030]

    cond = max_ach != 1

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    # Need this backend for figures to be compatible with CorelDraw
    with PdfPages('spike_prob.pdf') as pdf:
        ax[0].plot(np.arange(nt)-10,mean_control_up,'k')
        ax[0].plot(np.arange(nt)-10,mean_gaba_up,'r')
        ax[0].plot(np.arange(nt)-10,mean_ach_up,'b')
        bottom,top = ax[0].get_ylim()
        ax[0].text(0,top-0.5,'n={}'.format(n_increase))
        ax[0].legend(['Control','GABAzine','Muscarine'])
        ax[0].set_xlabel('time (ms)')
        ax[0].set_ylabel('spike probability (norm.)')

        ax[1].plot(np.arange(nt)-10,mean_control_down,'k')
        ax[1].plot(np.arange(nt)-10,mean_gaba_down,'r')
        ax[1].plot(np.arange(nt)-10,mean_ach_down,'b')
        bottom,top = ax[1].get_ylim()
        ax[1].text(0,top-0.2,'n={}'.format(n_decrease))
        ax[1].set_xlabel('time (ms)')

        x = np.max(control[cond],axis=1)
        y = np.max(ach[cond], axis=1)
        c = Counter(zip(x,y))
        s = [2*c[(i,j)] for i,j in zip(x,y)]
        ax[2].scatter(x,y,c='k',s=s)
        ax[2].set_xlabel('Control peak spike probability')
        ax[2].set_ylabel('Muscarine peak spike probability')
        start,stop = ax[2].get_xlim()
        x = np.linspace(start,stop)
        y = x
        ax[2].plot(x,y,'--k')
        pdf.savefig()
        plt.close()

###############################################################################
# Code to construct connection summary figure;                                #
# Panel 1: Max. probability in muscarine vs. number active MF inputs.         #
# Panel 2: Max. probability in muscarine vs. Avg. net conductance             #
# Panel 3: Max. probability in muscarine vs. Fraction reduction in tonic inh. #
###############################################################################
def plot_connections_summary(counts, max_ach, int_g_e, int_g_i, reduce_tonic):
    avg_g_e = int_g_e/60
    avg_g_i = int_g_i/60
    avg_g_sub = avg_g_e - avg_g_i

    cond = (max_ach != 1) & (max_ach != 0)
    nospikes = max_ach == 0
    # Generate figure object
    fig,ax = plt.subplots(1, 3, figsize=(12,4))
    with PdfPages('compare_connections.pdf') as pdf:
        # Marker size proportional to number of instances
        c = Counter(zip(counts[cond],max_ach[cond]))
        s = [2*c[(i,j)] for i,j in zip(counts[cond],max_ach[cond])]
        cc = Counter(zip(counts[nospikes],max_ach[nospikes]))
        ss = [2*cc[(i,j)] for i,j in zip(counts[nospikes],max_ach[nospikes])]
        # Plot maximum spike probability in muscarine against number of active inputs
        ax[0].scatter(counts[cond],np.log(max_ach[cond]),c='k',s=s)
        ax[0].scatter(counts[nospikes],np.full(sum(nospikes),-2),c='k',s=ss)
        ax[0].set_xticks(np.arange(0,5,1))
        ax[0].set_xlabel('No. active MF inputs')
        ax[0].set_ylabel('Log max. spike prob. in muscarine (norm)')

        ax[1].scatter(avg_g_sub[cond], np.log(max_ach[cond]), c='k')
        ax[1].scatter(avg_g_sub[nospikes],np.full(sum(nospikes),-2),c='k')
        ax[1].set_xlabel('Avg. net conductance (nS)')

        # Plot maximum spike probability in muscarine against reduction in tonic inhib.
        ax[2].scatter((1-reduce_tonic[cond]),np.log(max_ach[cond]),c='k')
        ax[2].scatter((1-reduce_tonic[nospikes]),np.full(sum(nospikes),-2),c='k')
        ax[2].set_xlabel('Fractional reduction in tonic inhibition')
        ax[2].set_xlim(0,1)

        pdf.savefig()
        plt.close()

################################################################################
# Code to construct figure to visualize inh. vs. exc. conductance, with points #
# color coded based on max. spike probability in muscarine                     #
################################################################################
def plot_conductance_scatter_heatmap(int_g_e, int_g_i, max_ach):
    # Don't plot cells whose spike probability stayed the same
    cond = (max_ach != 1) & (max_ach != 0)
    nospikes = max_ach == 0
    avg_g_e = int_g_e/60
    avg_g_i = int_g_i/60
    colors = np.log(max_ach[cond])
    orig_colormap = plt.cm.coolwarm
    vmax = np.max(colors)
    vmin = np.min(colors)
    new_mid = 1-vmax/(vmax+abs(vmin))
    shifted_cmap = shiftedColorMap(orig_colormap,midpoint=new_mid,name='shifted')
    plt.figure()
    with PdfPages('conductance_scatter_heatmap.pdf') as pdf:
        plt.scatter(avg_g_e[nospikes],avg_g_i[nospikes],c='k')
        sc = plt.scatter(avg_g_e[cond],avg_g_i[cond],c=colors,cmap=shifted_cmap)
        cb = plt.colorbar(sc)
        cb.set_label('Log max. spike prob in muscarine (norm.)',rotation=270)
        plt.xlim(-0.5,6)
        plt.ylim(-0.5,6)
        plt.xlabel('Avg. excitatory conductance (nS)')
        plt.ylabel('Avg. inhibitory conductnace (nS)')
        pdf.savefig()
        plt.close()

def plot_input_distributions(int_g_e, int_g_i,reduce_tonic, max_ach):
    increase = max_ach > 1
    decrease = max_ach < 1
    avg_g_e = int_g_e/60
    avg_g_i = int_g_i/60
    avg_g_net = avg_g_e-avg_g_i
    max_g_e = np.max(avg_g_e)
    max_g_i = np.max(avg_g_i)
    max_g_net = np.max(avg_g_net)
    min_g_net = np.min(avg_g_net)
    tonic_reduction = 1-reduce_tonic
    max_tonic = np.max(tonic_reduction)
    min_tonic = np.min(tonic_reduction)
    binsize = 0.1
    binmax  = max(max_g_e,max_g_i)
    bins    = np.arange(0,binmax,binsize)
    bins_net = np.arange(min_g_net,max_g_net,binsize)
    bins_tonic = np.arange(min_tonic,max_tonic,binsize*0.1)
    fig,ax = plt.subplots(2,2,figsize=(12,8))
    with PdfPages('input_distributions.pdf') as pdf:
        ax[0,0].hist(avg_g_e[increase],bins=bins,density=True,histtype='step',color='r',label='increase')
        ax[0,0].hist(avg_g_e[decrease],bins=bins,density=True,histtype='step',color='b',label='decrease')
        ax[0,0].set_xlabel('Avg. excitatory conductance (nS)')
        ax[0,0].set_ylabel('Density')
        ax[0,0].legend(loc='upper right')

        ax[0,1].hist(avg_g_i[increase],bins=bins,density=True,histtype='step',color='r')
        ax[0,1].hist(avg_g_i[decrease],bins=bins,density=True,histtype='step',color='b')
        ax[0,1].set_xlabel('Avg. inhibitory conductance (nS)')
        ax[0,1].set_ylabel('Density')

        ax[1,0].hist(avg_g_net[increase],bins=bins_net,density=True,histtype='step',color='r')
        ax[1,0].hist(avg_g_net[decrease],bins=bins_net,density=True,histtype='step',color='b')
        ax[1,0].set_xlabel('Avg. net conductance (nS)')
        ax[1,0].set_ylabel('Density')

        ax[1,1].hist(tonic_reduction[increase],bins=bins_tonic,density=True,histtype='step',color='r')
        ax[1,1].hist(tonic_reduction[decrease],bins=bins_tonic,density=True,histtype='step',color='b')
        ax[1,1].set_xlabel('Frac. reduction in tonic inhibition')
        ax[1,1].set_ylabel('Density')
        pdf.savefig()
        plt.close()

def plot_kde_inputs(int_g_e,int_g_i,reduce_tonic, max_ach):
    increase = max_ach > 1
    decrease = max_ach < 1
    avg_g_e = int_g_e/60
    avg_g_i = int_g_i/60
    avg_g_net = avg_g_e-avg_g_i
    tonic_reduction = 1-reduce_tonic
    max_g_e = np.max(avg_g_e)
    max_g_i = np.max(avg_g_i)
    binmax = max(max_g_e,max_g_i)+1
    binmin_net = np.min(avg_g_net)-1
    binmax_net = np.max(avg_g_net)+1
    bandwidth = 0.1
    X_plot = np.linspace(-1,binmax,int(2*binmax/bandwidth))[:,np.newaxis]
    X_plot_net = np.linspace(binmin_net,binmax_net,int(2*binmax/bandwidth))[:,np.newaxis]
    bandwidth_tonic = 0.01
    X_plot_tonic = np.linspace(0,1,int(2*1/bandwidth_tonic))[:,np.newaxis]

    kde_increase_e = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_e[increase,np.newaxis])
    kde_decrease_e = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_e[decrease,np.newaxis])
    kde_increase_i = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_i[increase,np.newaxis])
    kde_decrease_i = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_i[decrease,np.newaxis])
    kde_increase_net = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_net[increase,np.newaxis])
    kde_decrease_net = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(avg_g_net[decrease,np.newaxis])
    kde_increase_tonic = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(tonic_reduction[increase,np.newaxis])
    kde_decrease_tonic = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(tonic_reduction[decrease,np.newaxis])

    logdens_increase_e = kde_increase_e.score_samples(X_plot)
    logdens_decrease_e = kde_decrease_e.score_samples(X_plot)
    logdens_increase_i = kde_increase_i.score_samples(X_plot)
    logdens_decrease_i = kde_decrease_i.score_samples(X_plot)
    logdens_increase_net = kde_increase_net.score_samples(X_plot_net)
    logdens_decrease_net = kde_decrease_net.score_samples(X_plot_net)
    logdens_increase_tonic = kde_increase_tonic.score_samples(X_plot_tonic)
    logdens_decrease_tonic = kde_decrease_tonic.score_samples(X_plot_tonic)

    fig,ax = plt.subplots(2,2,figsize=(12,8))
    with PdfPages('inputs_kde.pdf') as pdf:
        ax[0,0].plot(X_plot[:,0],np.exp(logdens_increase_e),'-r',label='increase')
        ax[0,0].plot(X_plot[:,0],np.exp(logdens_decrease_e),'-b',label='decrease')
        ax[0,0].set_xlim(0,binmax-1)
        ax[0,0].legend(loc='upper right')
        ax[0,0].set_xlabel('Avg. exc. conductance (nS)')
        ax[0,0].set_ylabel('Density')

        ax[0,1].plot(X_plot[:,0],np.exp(logdens_increase_i),'-r')
        ax[0,1].plot(X_plot[:,0],np.exp(logdens_decrease_i),'-b')
        ax[0,1].set_xlim(0,binmax-1)
        ax[0,1].set_xlabel('Avg. inh. conductance (nS)')
        ax[0,1].set_ylabel('Density')

        ax[1,0].plot(X_plot_net[:,0],np.exp(logdens_increase_net),'-r')
        ax[1,0].plot(X_plot_net[:,0],np.exp(logdens_decrease_net),'-b')
        ax[1,0].set_xlim(binmin_net,binmax_net)
        ax[1,0].set_xlabel('Avg. net conductance (nS)')
        ax[1,0].set_ylabel('Density')

        ax[1,1].plot(X_plot_tonic[:,0],np.exp(logdens_increase_tonic),'-r')
        ax[1,1].plot(X_plot_tonic[:,0],np.exp(logdens_decrease_tonic),'-b')
        ax[1,1].set_xlim(0,1)
        ax[1,1].set_xlabel('Frac. reduction in tonic inhibition')
        ax[1,1].set_ylabel('Density')
        pdf.savefig()
        plt.close()

def plot_inputs_summary(int_g_e, int_g_i, max_ach, reduce_tonic):
    
    exc = int_g_e/60
    inh = int_g_i/60
    net = exc-inh
    increase = max_ach > 1
    decrease = max_ach < 1

    mean_inc_e = np.mean(exc[increase])
    sem_inc_e  = np.std(exc[increase])/np.sqrt(sum(increase))
    mean_dec_e = np.mean(exc[decrease])
    sem_dec_e  = np.std(exc[decrease])/np.sqrt(sum(decrease))

    mean_inc_i = np.mean(inh[increase])
    sem_inc_i  = np.std(inh[increase])/np.sqrt(sum(increase))
    mean_dec_i = np.mean(inh[decrease])
    sem_dec_i  = np.mean(inh[decrease])/np.sqrt(sum(decrease))

    mean_inc_net = np.mean(net[increase])
    sem_inc_net  = np.std(net[increase])/np.sqrt(sum(increase))
    mean_dec_net = np.mean(net[decrease])
    sem_dec_net  = np.std(net[decrease])/np.sqrt(sum(decrease))

    reduce = 1-reduce_tonic
    mean_inc_reduce = np.mean(reduce[increase])
    sem_inc_reduce  = np.std(reduce[increase])/np.sqrt(sum(increase))
    mean_dec_reduce = np.mean(reduce[decrease])
    sem_dec_reduce  = np.std(reduce[decrease])/np.sqrt(sum(decrease))


    fig,ax = plt.subplots(2, 2, figsize=(12,4))
    with PdfPages('inputs_summary.pdf') as pdf:
        # ax[0,0].scatter([1,2],[mean_dec_e,mean_inc_e],c='k')
        ax[0,0].errorbar([1,2],[mean_dec_e,mean_inc_e],[sem_dec_e,sem_inc_e],fmt='ok',ecolor='k')
        ax[0,0].set_xticks(np.arange(0,3,1))
        ax[0,0].set_xticklabels(['','Decreasing','Increasing',''])
        ax[0,0].set_ylabel('ge (nS)')
        ax[0,0].set_xlim(0,3)

        # ax[0,1].scatter([1,2],[mean_dec_i,mean_inc_i],c='k')
        ax[0,1].errorbar([1,2],[mean_dec_i,mean_inc_i],[sem_dec_i,sem_inc_i],fmt='ok',ecolor='k')
        ax[0,1].set_xticks(np.arange(0,3,1))
        ax[0,1].set_xticklabels(['','Decreasing','Increasing',''])
        ax[0,1].set_ylabel('gi (nS)')
        ax[0,1].set_xlim(0,3)

        # ax[1,0].scatter([1,2],[mean_dec_net,mean_inc_net],c='k')
        ax[1,0].errorbar([1,2],[mean_dec_net,mean_inc_net],[sem_dec_net,sem_inc_net],fmt='ok',ecolor='k')
        ax[1,0].set_xticks(np.arange(0,3,1))
        ax[1,0].set_xticklabels(['','Decreasing','Increasing',''])
        ax[1,0].set_ylabel('gnet (nS)')
        ax[1,0].set_xlim(0,3)

        # ax[1,1].scatter([1,2],[mean_dec_reduce,mean_inc_reduce],c='k')
        ax[1,1].errorbar([1,2],[mean_dec_reduce,mean_inc_reduce],[sem_dec_reduce,sem_inc_reduce],fmt='ok',ecolor='k')
        ax[1,1].set_xticks(np.arange(0,3,1))
        ax[1,1].set_xticklabels(['','Decreasing','Increasing',''])
        ax[1,1].set_ylabel('gt reduction')
        ax[1,1].set_xlim(0,3)
        ax[1,1].set_ylim(0,1)
        pdf.savefig()
        plt.close()

def shiftedColorMap(cmap,start=0,midpoint=0.5,stop=1.0, name='shiftedcmap'):
    cdict = {
    'red': [],
    'green': [],
    'blue': [],
    'alpha': []
    }
    # Regular index to compute colors
    reg_index = np.linspace(start,stop,257)

    # Shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0,midpoint,128,endpoint=False),
                             np.linspace(midpoint,1.0,129,endpoint=True)])

    for ri,si in zip(reg_index,shift_index):
        r,g,b,a = cmap(ri)
        cdict['red'].append((si,r,r))
        cdict['green'].append((si,g,g))
        cdict['blue'].append((si,b,b,))
        cdict['alpha'].append((si,a,a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name,cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

if __name__ == '__main__':
    main()
