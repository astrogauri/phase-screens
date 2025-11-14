import numpy as np
import matplotlib.pyplot as plt

#%%
def pli(image):
    return plt.imshow(image.T,origin="lower")

#%%
def covariance_function(phase):
    '''Computes the covariance of the phase derivative from a given simulated phase screen. 
       Covariance of the phase derivative is related to the phase structure function.'''

    n = 100
    # Cropping phase screen 
    phase_derivative_crop = phase[0:n, 0:n] - phase[1:n+1, 0:n]


    # Computing autocorrelation for the phase screen
    ext_phase_screen = np.zeros((2*n,2*n))
    ext_phase_screen[0:n,0:n] = phase_derivative_crop
    auto = (np.fft.ifft2(np.abs(np.fft.fft2(ext_phase_screen))**2)).real
    auto = np.fft.fftshift(auto)
    auto = auto[n//2:n//2+n, n//2:n//2+n] # To use central region of autocorrelation to avoid 0 division

    # Computing normalization 
    support = np.zeros((2*n,2*n))
    support[0:n,0:n] = 1
    normalization = (np.fft.ifft2(np.abs(np.fft.fft2(support))**2)).real
    normalization = np.fft.fftshift(normalization)
    normalization = normalization[n//2:n//2+n, n//2:n//2+n]

   # Applying normalization
    auto_normalized = auto / normalization

    # Computing covariance of derivative of phase
    covariance =  auto_normalized

    return phase_derivative_crop, auto, normalization, covariance

#%%
# Simulated phase screen for testing. We take multiple phase screens to average the covariance.
phase_screen_array = []
no_screens = 10

# Generating multiple phase screens
for i in range(no_screens):
    N=200
    x = np.arange(N)-N//2
    k2 = np.fft.fftshift(x[None,:]**2 + x[:,None]**2) + 1.0
    phase_img = np.fft.fft2(k2**(-11/12) * np.exp(2j*np.pi*np.random.rand(N,N))).real 
    phase_screen_array.append(phase_img)


# Loop over all phase screens and compute average covariance
cov = 0.0
for i in range(no_screens):
    _, _, _, covariance = covariance_function(phase_screen_array[i])
    cov = cov + covariance
cov = cov / no_screens

pli(cov)

#%%
# Scatter plot of r vs covariance
n = 100
tmp = np.arange(n) - n//2
X, Y = np.meshgrid(tmp, tmp, indexing='ij')
r = np.sqrt(X**2 + Y**2)

covariance_theoretical = 5/(3*r**(1/3))-5*X**2/(9*r**(7/3))

plt.figure()
#plt.scatter(r,cov.flatten() , s=1, label='experimental')
#plt.scatter(r, covariance_theoretical.flatten()/150, s=1, color='r', label='theoretical')
plt.scatter(cov.flatten(),covariance_theoretical.flatten(), s=1)
#plt.xlabel("Radial Distance r")
#plt.ylabel("Covariance of Phase Derivative")
plt.legend()
plt.show()

