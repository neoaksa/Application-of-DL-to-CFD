#!/usr/bin/python
# Comflics: Exploring OpenFOAM
# Compute Strouhal Number of Laminar Vortex Shedding
# S. Huq, 13MAY17
#
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# # Read Results
data = np.loadtxt('./postProcessing/forceCoeffs/0/forceCoeffs.dat', skiprows=0)

L       = 2           # L = D - Diameter
V       = 1           # Velocity
time    = data[:,0]
Cd      = data[:,2]
Cl      = data[:,3]

del data

# # Compute FFT

N       = len(time)
dt      = time[2] - time[1]

# # inaccurate FFT
# freq    = np.fft.fftfreq(N, dt)
# Cd_fft  = np.fft.fft(Cd) 
# Cl_amp  = np.fft.fft(Cl) 
# plt.plot(freq, Cl_amp)       # Figure 2.10
# plt.show()

# # Better stable FFT
nmax=512                       # no. of points in the fft
# freq, Cd_amp = signal.welch(Cd, 1./dt, nperseg=nmax)
freq, Cl_amp = signal.welch(Cl, 1./dt, nperseg=nmax)
plt.plot(freq, Cl_amp)         # Figure 2.10
plt.show() 

# # Strouhal Number
# Find the index corresponding to max amplitude
Cl_max_fft_idx = np.argmax(abs(Cl_amp))  
freq_shed      = freq[Cl_max_fft_idx ]
St             = freq_shed * L / V

print "Vortex shedding freq: %.3f [Hz]" % (freq_shed)
print "Strouhal Number: %.3f" % (St)

# # Explore Results
# # 
# # Figure 2.8
# # See if there atleast 10 cycles of oscillation
# # improves the accuracy; 
# plt.plot(time,Cl)
# plt.show()
# # Figure 2.9
# plt.plot(time,Cd)
# plt.show()
# # 
# # Exercise
# # Exclude data before onset of the oscillations. 
# # approx time = 200 s.
# # Hint: skiprows = 800 - 950


