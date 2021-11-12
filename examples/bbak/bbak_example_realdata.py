from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import sys
sys.path.append('./../../')
import DLM_tools

plt.rcParams['font.size'] = 14.

# import data (omit samples, manipulate length, if needed) 
data = np.genfromtxt('test_data_bbak_2020-06-15.csv')
#data = data[32000::2] # make sure to adjust the period p upon sub-sampling
n = len(data)
grid = np.arange(n)

# show data
fig,ax = plt.subplots()
ax.plot(np.arange(n),data,'.')
ax.set_title('Time series (Doc Bak data)')
ax.set_xlabel('Sample no.')

# initialize DLM
p = 250 # period
sigma = DLM_tools.SigmaEvolutionTuple(level=1e-1, harmonic=1e-1, obs_noise=1e-1) # tuning parameters
my_dlm = DLM_tools.Local_Level_Single_Harmonic_DLM(p=p, sigma=sigma)

# operate DLM across the 'data'
estimated_state = np.zeros([n,3])
forecasts = np.zeros([n,2*p]) # do forecasting two periods ahead
print('Iterating DLM')
for i in tqdm(range(n),ascii=True):
    my_dlm.forward_filter(Yt=data[i])
    estimated_state[i,:] = my_dlm.get_state()
    forecasts[i,:] = my_dlm.forecast(k=np.arange(1,2*p+1))

# plot estimated model components across time
level = estimated_state[:,0]
harmonic = estimated_state[:,1] # zero-mean cyclic component
fig,ax = plt.subplots(nrows=2, sharex=True, sharey=True)
ax[0].plot(grid, data, '.')
ax[0].plot(grid, level,'k', label='DLM level')
ax[1].plot(grid, data,'.')
ax[1].plot(grid, level + harmonic,'r', label='DLM level + DLM (zero-mean) harmonic')
ax[0].set_title('Time series (Doc Bak data)')
ax[1].set_xlabel('Sample no.')
ax[0].legend(loc=0)
ax[1].legend(loc=0)

"""
# make simple animation to show forecasting capability
fig,ax = plt.subplots()
ax.set_title('Forecasting visualization')
ax.set_xlabel('Sample no.')
ax.plot(grid, data, '.', markersize=3.0)
line, = ax.plot([], [], color='r')
point, = ax.plot([], [], color='k', marker='o')
ax.legend(['Data','Two-period DLM forecast', 'Present'], fontsize=10)

def update_func(i, x, y, line, point, ax, p=2*p): # helper function
    line.set_data(x[i+1:i+1+p], y[i,:]) # y is the (n x 2p) forecast array
    point.set_data(x[i], y[i,0])
    ax.set_xlim([x[i]-1.5*p, x[i+p]+1.5*p])
    ax.set_ylim([y[i,:].min()-0.4, y[i,:].max()+0.4])
    return line, # trailing comma here is crucial if blit=True

print('Creating animation')
my_movie = ani.FuncAnimation(fig, 
                             update_func, 
                             frames = n//4, # for i in range(frames)  
                             fargs = (grid,forecasts,line,point,ax),
                             interval = 10, # [ms]
                             blit = False)


if 0:
    print('Saving animation (slow)')
    my_movie.save('forecasting.mp4')
"""

# 
plt.show(block=False)

#