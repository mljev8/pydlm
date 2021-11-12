"""
Specialized DLM constructions, standard models, toy examples, etc.
"""

import numpy as np
from scipy.linalg import block_diag
from collections import namedtuple

from DLM_core import DLM_abc, DLM_default


SigmaEvolutionTuple = namedtuple('SigmaEvolutionTuple', 
                                 ['level','harmonic','obs_noise'], defaults=3*[1e-1])

class Local_Level_Single_Harmonic_DLM(DLM_default):
    """
    Doc Bak's use case
    Local level with a zero-mean harmonic component on top
    Thus, the latent evolution model has 3 time-varying parameters (n=3)
    The discrete period p is considered fixed, i.e. it's not a dynamic parameter
    """
    def __init__(self, p, lamb=1., sigma=SigmaEvolutionTuple()):
        assert isinstance(p, int) and p >= 2, "Period p must be >= 2"
        DLM_default.__init__(self, n=1+2, r=1)

        self._p = p
        self._lambda = lamb
        self._sigma = sigma
        self._omega = 2.*np.pi*(1./p) # convert period p to discrete frequency
        cos, sin = np.cos(self._omega), np.sin(self._omega)
        
        G1 = np.eye(1)
        W1 = np.eye(1) * sigma.level**2
        G2 = self._lambda * np.array([[cos,sin],[-sin,cos]]) # 2x2 rotation matrix
        W2 = np.eye(2) * sigma.harmonic**2                   # 2x2
        
        G = block_diag(G1,G2) # 3x3
        F = np.c_[1.,1.,0.].T # 3x1
        W = block_diag(W1,W2) # 3x3
        V = np.c_[sigma.obs_noise**2] # scalar
        
        self.init_dlm(G,F,W,V)
        return

    def forecast(self, k):
        level = self._m[0]
        cos_part = self._m[1] * np.cos(k*self._omega)
        sin_part = self._m[2] * np.sin(k*self._omega)
        return level + (cos_part + sin_part) * self._lambda**k

#
class Polynomial_Trend_DLM(DLM_default):
    def __init__(self, n):
        assert isinstance(n, int) and n >= 1, "Parameter n must be >= 1"
        DLM_default.__init__(self, n=n, r=1)
        
        G = np.eye(n)
        for i in range(n-1):
            G[i,i+1] = 1.
        # need to implement the rest at some point

#
class Harmonic_Component_DLM(DLM_default):
    def __init__(self, p, lamb=1.): # [W&H] Sec. 5.5.4, 6.1.2, 8.6.3
        assert isinstance(p, int) and p >= 2, "Period p must be >= 2"
        DLM_default.__init__(self, n=2, r=1)
        
        self._lambda = lamb
        self._omega = 2.*np.pi*(1./p)
        cos, sin = np.cos(omega), np.sin(self._omega)
        G = self._lambda * np.array([[cos,sin],[-sin,cos]]) # 2x2
        # need to implement the rest at some point

#
class Form_Free_Seasonal_DLM(DLM_default):
    def __init__(self, p): # [W&H] Sec. 8.3, 8.4 and 8.5
        assert isinstance(p, int) and p >= 2, "Period p must be >= 2"
        DLM_default.__init__(self, n=p, r=1)
        
        G = np.roll(np.eye(n), shift=1, axis=0) # pop rotation
        # need to implement the rest at some point

#
class Fourier_Form_Seasonal_DLM(DLM_default):
    def __init__(self, p): # [W&H] Sec. 8.6
        assert isinstance(p, int) and p >= 2, "Period p must be >= 2"
        DLM_default.__init__(self, n=p, r=1)
        # need to implement the rest at some point

#
class ARMA_DLM(DLM_default):
    def __init__(self, p, q): # [W&H] Sec. 9.4.6
        assert isinstance(p, int) and isinstance(q, int), "p and q must be integers"
        assert min(p,q) >= 0 and max(p,q) >= 1, "p and q must be non-negative"
        DLM_default.__init__(self, n=max(p,q+1), r=1)        
        # need to implement the rest at some point

#
