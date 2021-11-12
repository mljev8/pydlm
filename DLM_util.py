"""
DLM helper functionalities
"""

import numpy as np
from scipy.special import factorial # fix this at some point
from scipy.linalg import block_diag

#
class DLM_particle(object):
    """
    Linear model of 1D particle motion with random excitation.
    Key reference [EVS], E. V. Stansfield, Introduction To Kalman Filters.
    """    
    
    @staticmethod
    def particle_transition(tau, n=3): # Transition matrix (evolution matrix)
        assert tau > 0., "Time-step parameter must be strictly positive"
        G = np.zeros([n,n])
        row = float(tau)**np.arange(n) # powers of tau
        row /= factorial( np.arange(n) ) # divide by factorials
        G[0,:] = row[:] # first row
        for i in range(n-1):
            G[i+1,:] = np.r_[0.,G[i,0:-1]] # pop last entry of previous row
        return G
    
    @staticmethod
    def particle_covar(tau, sigma, n=3): # evolution covariance matrix
        assert tau > 0., "Time-step parameter must be strictly positive"
        powers = np.asfarray(np.arange(n)[::-1]) # [n-1,n-2,...,1,0]
        W = np.outer(float(tau)**powers, float(tau)**powers)
        W /= np.outer( factorial(powers), factorial(powers) )
        sum_term = np.ones([n,n])
        sum_term += np.reshape(np.kron(powers,np.ones(n)),[n,n])
        sum_term += np.reshape(np.kron(np.ones(n),powers),[n,n])
        W /= sum_term
        return float(tau*sigma**2) * W
#DLM_particle
