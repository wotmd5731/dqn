# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:35:42 2018

@author: JAE
"""

    ε: 0<ε<1, Percent of time to use exploration
    S, A: Sets of states and actions
    α: Learning rate
    Ν: Number of steps
    Tmax: Number of total steps
    I: Update target network period
    θ: parameters of global network
t <-- 0
θ_target <-- θ
Repeat:
    # parameters of local network
    θ_local <-- θ
    tstart <-- t
    previous_rewards <-- []
    previous_states <-- []
    previous_actions <-- []
    Get state s
    Repeat:
        Choose random e, 0 < e < 1
        If e < ε
            a <-- random.choose(As)
        Else
            a <-- argmaxQ(s, a, Θ_local)
                            
        Take action a and observe r and s'
        previous_rewards[i] <-- r
        previous_states[i] <-- s
        previous_actions[i] <-- a
        t <-- t+1
        T <-- T+1
        s <-- s'
    If t mod I == 0
        θ_target <-- θ
        
    Until (s is terminal) or (t-tstart==N)
   
    If s is terminal
        R <-- 0
    Else
        R <-- max(Q(s,a,θ_target))
        
    # compute loss 
    L <-- 0
    For i = t-1 : tstart
        r_i <--  previous_rewards[i]
        s_i <--  previous_states[i]
        a_i <--  previous_actions[i]
        R <-- r_i + γR      
        L <-- L + (R - Q(s_i,a_i, Θ_local))^2
    θ <-- optimize(L)
Until T>Tmax 