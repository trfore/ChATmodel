#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:52:52 2019

@author: trf
"""
from brian2 import * 
import brian2_cb_modeling2 as cbmodel #Classes that accept more than 1 'drug' condition
import pandas as pd

import os
import time

start_scope()
#==============================================================================
# User Input Section
#==============================================================================
### Simulation Flags
Use_Random_Seed = True                      # To seed or not to seed that's ?
Seed_Value      = 1492                      # Random Seed Value

Trial_duration  =  100                      # Trial duration in ms
defaultclock.dt =    0.01 * ms              # Integration time step
    
TMsynapes       = True                       # Logical, True = Tsodyks & Markram
                                             #  syn eq; False = Valera & Abbott. 
                                             
# GrC Parameters
cell_refract   =   2.0    * ms               # refractory period after spike
gl_GrC         =   0.789  * nS               # leak conductance, calc: 0.7886
                                             #  w/ Erev at -80
gl_Tonic       =   0.1    * nS               # tonic conductance
El_GrC         = -80      * mV               # reversal potential
rm_GrC         = -80      * mV               # initial resting potential for GrC
Cm_GrC         =   3      * pF               # membrane capacitance for GrC
v_th_GrC       = -47      * mV               # threshold potential for GrC 
v_reset_GrC    = -80      * mV               # reset potential for GrC

# GoC Parameters
#TODO
gl_GoC         =   0.789  * nS               # leak conductance, calc: 0.7886
                                             #  w/ Erev at -80
gl_Tonic       =   0.1    * nS               # tonic conductance
El_GoC         = -80      * mV               # reversal potential
rm_GoC         = -65      * mV               # initial resting potential
Cm_GoC         =  30      * pF               # membrane capacitance 
v_th_GoC       = -55      * mV               # threshold potential 
v_reset_GoC    = -65      * mV               # reset potential 


E_Exc          =   0      * mV               # Excitatory reversal potential
E_Inh          = -80      * mV               # Inhibitory reversal potential

g_Exc          =   3.25   * nS               # Excitatory maximum conductance
g_Inh          =   2.0    * nS               # Inhibitory maximum conductance

t_Exc_decay    =   2.5    * ms               # Excitatory decay time constant
t_Exc_rise     =   0.15   * ms               # Excitatory rise time constant
t_Inh_decay    =   6.2    * ms               # Inhibitory decay 
t_Inh_rise     =   0.4    * ms               # Inhibitory rise

alpha_Exc      =   1      * ms**-1           # h0 scale factor
alpha_Inh      =   1      * ms**-1

#===============================================================================
### Neuron Equations
#===============================================================================
eq = Equations('''
dvm/dt = (I_GrC)/Cm_GrC                               : volt (unless refractory)    
I_GrC  = I_Exc + I_leak + I_tonic + I_Inh                       : amp
I_leak = gl_GrC*(El_GrC - vm)                                   : amp
I_tonic = gl_Tonic*(El_GrC - vm)                                : amp
I_Exc = g_Exc*(vm - E_Exc)* -y                                  : amp
I_Inh = g_Inh*(vm - E_Inh)* -z                                  : amp

#Exc two exp equation eq
dy/dt = -y / t_Exc_decay + alpha_Exc * h * (1 - y)              : 1 
dh/dt = -h / t_Exc_rise                                         : 1 

#Evoked Inh two exp equation eq
dz/dt = -z / t_Inh_decay + alpha_Inh * b * (1 - z)              : 1 
db/dt = -b / t_Inh_rise                                         : 1 
''')

cell = NeuronGroup(GrC_number+E_MF_number, model = eq, threshold = 'vm > v_th_GrC',
                     reset = 'vm = v_reset_GrC',
                     refractory = cell_refract, method='euler')
cell.vm = rm_GrC    # Init of membrane potential

#==============================================================================
### Synapse Equations
#==============================================================================
if TMsynapes is True:
    print('Synaptic STP Model: Using Tsodyks & Markram')
    
    # Tsodyks and Markram Synaptic Equations
    exc_synapses_eqs = '''
    # Usage of releasable neurotransmitter per single action potential:
    du_S/dt = -Omega_f * u_S                                  : 1 (clock-driven)
    # Fraction of synaptic neurotransmitter resources available:
    dx_S/dt = Omega_d *(1 - x_S)                              : 1 (clock-driven)
    wExc                                                      : 1
    '''
    exc_synapses_action = '''
    u_S += U_0 * (1 - u_S)
    r_S = u_S * x_S
    x_S -= r_S
    h_post += wExc * r_S
    '''
    inh_synapses_eqs = '''
    du_S/dt = -Omega_f_inh * u_S                              : 1 (clock-driven)
    dx_S/dt = Omega_d_inh *(1 - x_S)                          : 1 (clock-driven)
    dGoC_R/dt = (1 - GoC_R)/GoC_refract                       : 1 (clock-driven)
    wInh                                                      : 1
    randomProb :1
    '''
    inh_synapses_action = '''
    u_S += U_0_inh * (1 - u_S)
    r_S = u_S * x_S
    x_S -= r_S
    randomProb = rand()
    b_post += wInh * r_S * int(randomProb <= GoC_R)
    GoC_R = GoC_R * (GoC_R <= randomProb) 
    '''
elif TMsynapes is False:
    print('Synaptic STP Model: Using Valera & Abbott')
    
    # Valera/Chance/Abbott Synaptic Equations
    exc_synapses_eqs = Equations('''
    dDexc/dt = (1 - Dexc)/STD_tau                             : 1 (clock-driven)
    dFexc/dt = (1 - Fexc)/STF_tau                             : 1 (clock-driven)
    wExc                                                      : 1
    ''')
    exc_synapses_action  = '''
    h_post += wExc * Dexc   
    Dexc *= STD_scalar
    Fexc += STF_scalar                              
    '''
    inh_synapses_eqs = Equations('''
    dDinh/dt = (1 - Dinh)/STD_tau                             : 1 (clock-driven)
    dFinh/dt = (1 - Finh)/STF_tau                             : 1 (clock-driven)
    wInh                                                      : 1   
    ''')
    inh_synapses_action = '''
    b_post  += wInh * Dinh * E_I_scalar 
    Dinh *= STD_I_scalar
    Finh += STF_I_scalar
    '''
else:
    print('Warning Unknown Synaptic Equation Settings')
    
#===============================================================================
    
#===============================================================================
### Generate Synapses 
#===============================================================================
ExcSyn = Synapses(stimulus_input, cell, 
                  model= exc_synapses_eqs, 
                  on_pre = exc_synapses_action, 
                  method='euler', 
                  name='ExcSyn', 
                  multisynaptic_index= 'syn_num', 
                  delay= ExcSynDelay * ms)

InhSyn = Synapses(stimulus_input, cell, 
                  model= inh_synapses_eqs, 
                  on_pre = inh_synapses_action, 
                  method='euler', 
                  name='InhSyn',
                  multisynaptic_index= 'syn_num', 
                  delay= InhSynDelay * ms)

InhSponSyn = Synapses(sIPSCinput, cell, 
                      model= 'wInh : 1', 
                      on_pre= 'a_post += wInh', 
                      method='euler', 
                      name='sInhSyn', 
                      delay= 0 * ms)

#==============================================================================
### Synaptic Connections
#==============================================================================

#TODO

#===============================================================================
### Initialize Synapse Values
#===============================================================================
# Connections must be est. first!

if TMsynapes is True:
    excMonVars = ['u_S', 'x_S']    #StateMonitor Variables 
    inhMonVars = ['u_S', 'x_S', 'GoC_R', 'randomProb']
    
    ExcSyn.x_S = 1
    ExcSyn.run_regularly('''
                         x_S = 1
                         u_S = 0
                         ''', when='start', dt=Trial_duration * ms) 
    InhSyn.x_S = 1
    InhSyn.run_regularly('''
                         x_S = 1
                         u_S = 0
                         randomProb = 0
                         GoC_R = 1
                         ''', when='start', dt=Trial_duration * ms) 
    
    SynW_EPSC.x_S = 1
    SynW_EPSC.run_regularly('''
                            x_S = 1
                            u_S = 0
                            ''', when='start', dt=Trial_duration * ms) 
    InhSyn_IPSC.x_S = 1
    InhSyn_IPSC.run_regularly('''
                              x_S = 1
                              u_S = 0
                              ''', when='start', dt=Trial_duration * ms) 

elif TMsynapes is False:
    excMonVars = ['Dexc', 'Fexc']    # StateMonitor Variables
    inhMonVars = ['Dinh', 'Finh']
    
    ExcSyn.Dexc = 1
    ExcSyn.Fexc = 1
    InhSyn.Dinh = 1
    InhSyn.Finh = 1
    SynW_EPSC.Dexc = 1
    SynW_EPSC.Fexc = 1
    InhSyn_IPSC.Dinh = 1
    InhSyn_IPSC.Finh = 1