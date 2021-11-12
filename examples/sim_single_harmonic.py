import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./../')
import DLM_tools

n = 1017 # number of samples
p = 37 # period
sigma_observation_noise = 0.4

# simulate a fake realization
fake_time_series = np.zeros(n)
trend = np.zeros(n)
cosine = np.zeros(n)
for i in range(n):
    trend[i] = min(5., 0.05*(i-200)) if i > 200 else 0.
    cosine[i] = np.cos(2.*np.pi*(1./p)*i + np.pi/7) if i > 500 else 0.
fake_time_series[:] = trend[:] + cosine[:]
fake_time_series[:] += sigma_observation_noise * np.random.standard_normal(size=n)

plt.close('all')

fig,ax = plt.subplots()
ax.set_title('Time series (simulated)')
ax.plot(np.arange(n),fake_time_series,'.')
ax.set_xlabel('Sample no.')

# initialize DLM and run it across the faked data realization
sigma_dlm = DLM_tools.SigmaEvolutionTuple(level=1e-1, harmonic=1e-1, obs_noise=1e-1)
my_dlm = DLM_tools.Local_Level_Single_Harmonic_DLM(p=37, sigma=sigma_dlm)

estimated_state = np.zeros([3,n])
for i in range(n):
    my_dlm.forward_filter(Yt=fake_time_series[i])
    estimated_state[:,i] = my_dlm.get_state()

# plot estimated model components across time
fig,ax = plt.subplots(nrows=3)
ax[0].set_title('Time series (simulated) and DLM components')
for j in range(3):
    ax[j].plot(np.arange(n),fake_time_series,'.')
    ax[j].plot(np.arange(n),estimated_state[j,:],'r')
ax[-1].set_xlabel('Sample no.')

# overlay simulated truth and DLM reconstructions
fig,ax = plt.subplots(nrows=2)
ax[0].set_title('Overlay of simulated truth and estimated DLM components')
ax[0].plot(np.arange(n),trend,'k')
ax[0].plot(np.arange(n),estimated_state[0,:],'r')
ax[1].plot(np.arange(n),cosine,'k')
ax[1].plot(np.arange(n),estimated_state[1,:],'r')
ax[-1].set_xlabel('Sample no.')

plt.show()

