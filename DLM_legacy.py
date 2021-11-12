"""
Legacy implementations
"""

import numpy as np
from scipy.special import factorial # fix this at some point
from scipy.linalg import block_diag
from collections import namedtuple

#
class DLM_ParticleTool():
    """
    Various kinds of useful matrices and functionalities for DLM-based applications
    Particle motion (random excitation imposed e.g. on velocity or acceleration)
    The paper "E. V. Stansfield: Introduction To Kalman Filters" is recommendable
    """
    
    # EMPTY CONSTRUCTOR
    def __init__(self):
        pass
    
    # Particle-type transition matrix (evolution matrix)
    @staticmethod
    def particle_transition(tau, dimension=3):
        d = dimension # rename
        phi = np.zeros([d,d])
        first_row = float(tau)**np.arange(d) # powers of tau
        first_row /= factorial( np.arange(d) ) # divide by factorials
        phi[0,:] = first_row
        for i in range(d-1):
            phi[i+1,:] = np.r_[0,phi[i,0:-1]] # push away last entry
        return phi # (d x d)
    
    # Particle-type covariance matrix
    @staticmethod
    def particle_covar(tau, sigma, dimension=3):
        d = dimension # rename
        powers = np.asfarray(np.arange(d)[::-1]) # [d-1,d-2,...,1,0]
        W = np.outer(float(tau)**powers,float(tau)**powers)
        W /= np.outer( factorial(powers), factorial(powers) )
        sum_term = np.ones([d,d])
        sum_term += np.reshape(np.kron(powers,np.ones(d)),[d,d])
        sum_term += np.reshape(np.kron(np.ones(d),powers),[d,d])
        W /= sum_term
        return float(tau*sigma**2)*W # (d x d)
    
    # Diagonal scaling of square matrix (enforcement of stationarity)
    @staticmethod
    def scale_diagonal(a, dimension=3):
        d = dimension # rename
        S = np.ones([d,d])
        for i in range(d):
            S[i,i] *= a
        return S # (d x d)

#
class DLM_Class():
    """
    West & Harrison, Bayesian Forecasting and Dynamic Models, 2nd Edition, Springer, 1999
    The general theory, definitions, updating schemes, etc., are located in Chapter 4
    """
    
    # allocate/declare matrices and vectors
    def __init__(self, n, r, D=np.float64): # n = state dim, r = observation dim
        self.m = np.zeros(n, dtype=D) # state (n x 1)
        self.a = np.zeros(n, dtype=D) # prediction (n x 1), state
        self.f = np.zeros(r, dtype=D) # forecast (r x 1), measurement
        self.C = np.eye(n, dtype=D) # posterior covariance
        self.R = np.zeros([n, n], dtype=D) # prediction covariance
        self.Q = np.zeros([r, r], dtype=D) # forecast covariance
        self.A = np.zeros([n, r], dtype=D) # adaption matrix (Kalman gain)
        self.F = np.ones([n, r], dtype=D) # design matrix
        self.G = np.eye(n, dtype=D) # evolution matrix
        self.V = np.eye(r, dtype=D) # observation covariance
        self.W = np.eye(n, dtype=D) # evolution covariance
        self._n = n
        self._r = r
        self.RF = np.zeros([n, r], dtype=D) # R.dot(F) (n x r)

    # public methods    
    def init_State(self, m):
        self.m[:] = m[:]
    def init_State_Covar(self, C):
        self.C[:,:] = C[:,:]
    def init_Evolution_Matrix(self, G):
        self.G[:,:] = G[:,:]
    def init_Design_Matrix(self, F):
        self.F[:,:] = F[:,:]
    def init_Evolution_Covar(self, W):
        self.W[:,:] = W[:,:]
    def init_Measurement_Covar(self, V):
        self.V[:,:] = V[:,:]

    def get_State(self):
        return self.m[:] 
    def get_Covar(self):
        return self.C[:,:]
    
    def iterate(self, Yt):
        self.a[:]   = self.map_State(self.m)
        self.G[:,:] = self.jacobi_State(self.m)
        self.R[:,:] = self.evaluate_R(self.G, self.C, self.W)
        self.f[:]   = self.map_Measurement(self.a)
        self.F[:,:] = self.jacobi_Measurement(self.a)
        self.Q[:,:] = self.evaluate_Q(self.F, self.R, self.V)
        self.A[:,:] = self.evaluate_A(self.R, self.F, self.Q)
        self.m[:]   = self.a + self.A.dot(Yt - self.f) # master equation
        self.C[:,:] = self.evaluate_C(self.R, self.F, self.A)
        return
    
    # protected methods (override when necessary, e.g. for computational efficiency)
    def map_State(self, m):
        return self.G.dot(m)
        
    def jacobi_State(self, m):
        return self.G
    
    def map_Measurement(self, a):
        return self.F.T.dot(a)
        
    def jacobi_Measurement(self, a):
        return self.F
    
    def evaluate_R(self, G, C, W):
        return G.dot(C.dot(G.T)) + W
    
    def evaluate_Q(self, F, R, V):
        #self.RF[:,:] = R.dot(F)
        #return F.T.dot(self.RF) + V
        return F.T.dot(R.dot(F)) + V
    
    def evaluate_A(self, R, F, Q):
        #return self.RF.dot(np.matrix(self.Q).I))
        return R.dot(F.dot(np.matrix(self.Q).I))
    
    def evaluate_C(self, R, F, A):
        #return R - self.RF.dot(A.T)
        return R.dot(np.eye(self._n) - F.dot(A.T))
    
    def forecast(self, k): # f_t(k) = E[ Y_{t+k} | y_{1:t} ]
        G_power_k = self.G[:,:]
        for i in range(max(0,k-1)):
            G_power_k[:,:] = G_power_k.dot(self.G)
        return self.F.T.dot(G_power_k.dot(self.m))


#
SigmaEvolutionTuple = namedtuple('SigmaEvolutionTuple', 
                                 ['level','harmonic','obs_noise'], defaults=3*[1e-1])

class Local_Level_Single_Harmonic_DLM(DLM_Class):
    """
    Doc Bak's use case
    Local level with a zero-mean harmonic component on top
    Thus, the latent evolution model has 3 time-varying parameters (n=3)
    The discrete period p is considered fixed, i.e. it's not a dynamic parameter
    """
    def __init__(self, p, _lambda=1., sigma=SigmaEvolutionTuple()):
        DLM_Class.__init__(self, n=1+2, r=1)

        self._p = p
        self._lambda = _lambda
        self._sigma = sigma
        self._omega = 2.*np.pi*(1./p) # convert period p to discrete frequency
        cos, sin = np.cos(self._omega), np.sin(self._omega)
        
        G1 = np.eye(1)                  # 1x1
        W1 = np.eye(1) * sigma.level**2 # 1x1
        G2 = self._lambda * np.array([[cos,sin],[-sin,cos]]) # 2x2 rotation matrix
        W2 = np.eye(2) * sigma.harmonic**2                   # 2x2
        
        self.init_Evolution_Matrix( block_diag(G1,G2) ) # 3x3
        self.init_Design_Matrix( np.c_[1.,1.,0.].T )    # 3x1
        self.init_Evolution_Covar( block_diag(W1,W2) )  # 3x3
        self.init_Measurement_Covar(np.c_[sigma.obs_noise**2]) # scalar
        return

    def forecast(self, k): # override implementation in base class
        level = self.m[0]
        cos_part = self.m[1] * np.cos(k*self._omega)
        sin_part = self.m[2] * np.sin(k*self._omega)
        return level + (cos_part + sin_part) * self._lambda**k

#
class Polynomial_Trend_DLM(DLM_Class):
    def __init__(self, n):
        DLM_Class.__init__(self, n, 1)
        
        G = np.eye(n)
        for i in range(n-1):
            G[i,i+1] = 1.
        self.init_Evolution_Matrix(G)
        
        self.F[:] = 0.
        self.F[0] = 1.
        # need to implement the rest at some point

#
class Harmonic_Component_DLM(DLM_Class):
    def __init__(self, p, _lambda=1.): # [W&H] Sec. 5.5.4, 6.1.2, 8.6.3
        DLM_Class.__init__(self, 2, 1)
        
        self._lambda = _lambda
        self._omega = 2.*np.pi*(1./p)
        cos, sin = np.cos(omega), np.sin(self._omega)
        G = self._lambda * np.array([[cos,sin],[-sin,cos]])
        self.init_Evolution_Matrix(G)
        
        self.F[:] = 0.
        self.F[0] = 1.
        # need to implement the rest at some point

#
class Form_Free_Seasonal_DLM(DLM_Class):
    def __init__(self, p): # [W&H] Sec. 8.3, 8.4 and 8.5
        DLM_Class.__init__(self, n=p, r=1)
        
        G = np.roll(np.eye(n), shift=1, axis=0) # pop rotation
        self.init_Evolution_Matrix(G)
        # need to implement the rest at some point

#
class Fourier_Form_Seasonal_DLM(DLM_Class):
    def __init__(self, p): # [W&H] Sec. 8.6
        DLM_Class.__init__(self, n=p, r=1)
        # need to implement the rest at some point

#
class ARMA_DLM(DLM_Class):
    def __init__(self, p, q): # [W&H] Sec. 9.4.6
        DLM_Class.__init__(self, n=max(p,q+1), r=1)        
        # need to implement the rest at some point

#
