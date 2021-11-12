"""
Scratchpad area, nothing important 
"""

import numpy as np

# diff matrix H
n = 10
H = np.zeros([n-1,n], dtype='float')
for i in range(n-1):
    H[i,i:i+2] = np.r_[1,-1]

A = np.matrix( H.T.dot(H) )

B = np.matrix( H.T.dot(H) )
B[0,0] = 2
B[-1,-1] = 2

