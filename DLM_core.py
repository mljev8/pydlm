"""
DLM engine room
"""

from abc import ABC, abstractmethod
import numpy as np

#
class DLM_abc(ABC):
    """
    Most abstracted version of the DLM.
    Two crucial member attributes, _m and _C, considered private, with .get() methods.
    Implementations MUST be provided upon inheritance. Cannot instantiate as is.
    """
    def __init__(self, n: int, r: int, dtype=np.float64) -> None:
        assert min(n,r) >= 1
        self._n = n
        self._r = r
        self._dtype = dtype
        self._isinitialized = False
        self._m = np.zeros(n, dtype=dtype) # state (n x 1)
        self._C = np.eye(  n, dtype=dtype) # covariance (n x n) 
        return
    
    @abstractmethod
    def init_dlm(self) -> None:
        raise NotImplementedError("Must override init_dlm()")
        self._isinitialized = True
        return

    @abstractmethod
    def forward_filter(self, Yt: np.ndarray, Zt=None) -> None:
        """ filtering recursion, forward in time """
        self._m[:]   = self._filter_eval_m(Yt)
        self._C[:,:] = self._filter_eval_C()
        return
    
    @abstractmethod
    def backward_smooth(self, filtered_m: np.ndarray, filtered_C: np.ndarray) -> None:
        """ smoothing recursion, backward in time """
        self._m[:]   = self._smoother_eval_m(filtered_m, filtered_C)
        self._C[:,:] = self._smoother_eval_C(filtered_C)
        return
    
    def get_state(self) -> np.ndarray:
        return self._m[:] + 0.
        
    def get_covar(self) -> np.ndarray:
        return self._C[:,:] + 0.
#DLM_abc

#
class DLM_default(DLM_abc):
    """
    Default version of the DLM. Typical starting point. Mappings are linear.
    This class can be instantiated as is. No further overrides necessary.
    Yet, allmost any model/application requires further specialization.
    """
    
    def __init__(self, n, r, dtype=np.float64):
        
        # invoke constructor from (parent) abstract base class
        DLM_abc.__init__(self, n=n, r=r, dtype=dtype)        

        # model quadruple (G,F,W,V)
        self._G = np.eye(n, dtype=dtype) # evolution matrix
        self._F = np.ones([n, r], dtype=dtype) # design matrix (transposed)
        self._W = np.eye(n, dtype=dtype) # evolution covariance
        self._V = np.eye(r, dtype=dtype) # observation covariance
        
        # storage for intermediate calculations
        self._a  = np.zeros(n, dtype=dtype) # prediction (n x 1), state
        self._f  = np.zeros(r, dtype=dtype) # forecast (r x 1), observation
        self._R  = np.zeros([n, n], dtype=dtype) # prediction covariance
        self._Q  = np.zeros([r, r], dtype=dtype) # forecast covariance
        self._A  = np.zeros([n, r], dtype=dtype) # adaption matrix (Kalman gain)
        self._RF = np.zeros([n, r], dtype=dtype) # R.dot(F)
        self._B  = np.zeros([n, n], dtype=dtype) # smoother
        return

    def init_dlm(self, G, F, W, V):
        """ 
        Initialize model quadruple (G, F, W, V).
        Perform minimalistic checks (necessary conditions, not sufficient).
        Non-conformable dimensions will incur a deliberate crash.
        """
        assert np.abs(W-W.T).max() < 1e-6, "Matrix W doesn't seem like a healthy covar"
        assert np.abs(V-V.T).max() < 1e-6, "Matrix V doesn't seem like a healthy covar"
        self._G[:,:] = G[:,:]
        self._F[:,:] = F[:,:]
        self._W[:,:] = W[:,:]
        self._V[:,:] = V[:,:]
        self._isinitialized = True
        return
    
    def forward_filter(self, Yt, Zt=None):
        """ filtering recursion, carefully override participant routines as needed """
        assert (self._isinitialized == True), "Initialization of DLM required"
        # adjust model quadruple via auxiliary input Zt (optional)
        self._update_model(Zt, self._G, self._F, self._W, self._V)
        # perform intermediate calculations (state prediction)
        self._a[:] = self._map_state(self._m)
        self._jacobi_state(self._m, self._G)
        self._R[:,:] = self._eval_R()
        # apply intervention via auxiliary input Zt (optional)
        self._apply_intervention(Zt, self._a, self._R)
        # perform intermediate calculations (observation forecasting)
        self._f[:] = self._map_obs(self._a)
        self._jacobi_obs(self._a, self._F)
        self._Q[:,:] = self._eval_Q()
        # compute Kalman gain
        self._A[:,:] = self._eval_A()
        # perform key updates
        self._m[:]   = self._filter_eval_m(Yt)
        self._C[:,:] = self._filter_eval_C()
        return

    def backward_smooth(self, filtered_m, filtered_C):
        """ smoothing recursion afo. pairs of (m,C) from forward filtering """
        # perform intermediate calculations
        self._R[:,:] = self._G.dot(filtered_C.dot(self._G.T)) + self._W
        self._B[:,:] = filtered_C.dot(self._G.T.dot(np.matrix(self._R).I ))
        # perform key updates
        self._m[:]   = self._smoother_eval_m(filtered_m, filtered_C)
        self._C[:,:] = self._smoother_eval_C(filtered_C)
        return

#    def forecast(self, k: int):
#        return

#    def log_likelihood(self):
#        return
    
    @staticmethod
    def _update_model(Zt, G, F, W, V):
        """ method for updating the model quadruple (G, F, W, V) """
        pass
    
    def _map_state(self, m):
        return self._G.dot(m)
    
    @staticmethod    
    def _jacobi_state(m, G):
        """ the (n x n) derivative of map_state() """
        pass
    
    @staticmethod
    def _apply_intervention(Zt, a, R):
        """ method for applying intervention onto (a, R) """
        pass
    
    def _map_obs(self, a):
        return self._F.T.dot(a)

    @staticmethod
    def _jacobi_obs(a, F):
        """ the (n x r) derivative of map_obs() """
        pass
    
    def _eval_R(self):
        return self._G.dot(self._C.dot(self._G.T)) + self._W
    
    def _eval_Q(self):
        """ byproduct R.dot(F) used within eval_A() and eval_C() """
        #return self._F.T.dot(self._R.dot(self._F)) + self._V
        self._RF[:,:] = self._R.dot(self._F)
        return self._F.T.dot(self._RF) + self._V
    
    def _eval_A(self):
        #return self._R.dot(self._F.dot(np.matrix(self._Q).I))
        return self._RF.dot(np.matrix(self._Q).I)
    
    def _filter_eval_m(self, Yt):
        return self._a + self._A.dot(Yt - self._f)
    
    def _filter_eval_C(self):
        #return self._R.dot(np.eye(self._n) - self._F.dot(self._A.T))
        return self._R - self._RF.dot(self._A.T)
                
    def _smoother_eval_m(self, filtered_m, filtered_C):
        return filtered_m + self._B.dot(self._m - self._map_state(filtered_m))
    
    def _smoother_eval_C(self, filtered_C):
        return filtered_C + self._B.dot((self._C - self._R).dot(self._B.T))
#DLM_default
