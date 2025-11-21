import numpy as np
import matplotlib.pyplot as plt

#%%
def pli(image):
    return plt.imshow(image.T,origin="lower")

#%%
def phase_structure_function(phase):
    '''Computes the centered autocorrelation from a given simulated phase screen which is used to compare and verify if the phase is Kolmogorov'''

    n = 100
    # Cropping phase screen to remove edge effects (overlap in FFT)
    phase_crop = phase[0:n, 0:n]

    # Computing autocorrelation for the phase screen
    ext_phase_screen = np.zeros((2*n,2*n)) # Extending phase screen to avoid overlap in FFT
    ext_phase_screen[0:n,0:n] = phase_crop
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

    #% Why do we do this?
    # n//2 is the index used to get the center value of autocorrelation
    Dphi = 2*(auto_normalized[n//2,n//2] - auto_normalized) 

    return phase_crop, auto, normalization, auto_normalized, Dphi

#%%
n = 100

# Simulated phase screen for testing. 
phase_screen_array = []
no_screens = 10

# Generating multiple phase screens
for i in range(no_screens):
    N=200
    x = np.arange(N)-N//2
    k2 = np.fft.fftshift(x[None,:]**2 + x[:,None]**2) + 1.0
    phase_img = np.fft.fft2(k2**(-11/12) * np.exp(2j*np.pi*np.random.rand(N,N))).real 
    phase_screen_array.append(phase_img)
phase_screen_array = np.array(phase_screen_array)


# Loop over all phase screens and compute average covariance
Dphi_avg = 0.0
phase_screen_crop = 0.0
auto_correlation = 0.0
auto_norm = 0.0

for i in range(no_screens):
    phase_crop, auto, _, auto_normalized, Dphi = phase_structure_function(phase_screen_array[i])

    Dphi_avg = Dphi_avg + Dphi
    phase_screen_crop = phase_screen_crop + phase_crop
    auto_correlation = auto_correlation + auto 
    auto_norm = auto_norm + auto_normalized


Dphi_avg = Dphi_avg / no_screens
phase_screen_crop = phase_screen_crop / no_screens
auto_correlation = auto_correlation / no_screens
auto_norm = auto_norm / no_screens


#%%
# Plots avergaged phase screens and autocorrelations
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plt.subplot(2, 2, 1)
im1 = pli(phase_crop)
plt.title("Phase Screen")
plt.colorbar(im1)


plt.subplot(2, 2, 2)
im2 = pli(auto)
plt.title("Autocorrelation")
plt.colorbar(im2)


plt.subplot(2, 2, 3)
im3 = pli(auto_normalized)
plt.title("Autocorrelation (Normalized)")
plt.colorbar(im3)

plt.subplot(2, 2, 4)
im4 = pli(Dphi)
plt.title("Phase Structure Function D$\phi$")
plt.colorbar(im4)

plt.tight_layout()
plt.show()
#%%
# r is the radial distance from the center 
tmp = np.arange(n) - n//2
X, Y = np.meshgrid(tmp, tmp, indexing='ij')
r = np.sqrt(X**2 + Y**2)

# Cropping the autocorrelation and r 
Dphi_avg_crop = Dphi_avg[(n//4):(3*n//4), (n//4):(3*n//4)]
r_crop = r[(n//4):(3*n//4), (n//4):(3*n//4)]

# Comparing autocorrelation to r^(5/3)
k = np.sum(Dphi_avg_crop * r_crop**(5/3)) / np.sum(r_crop**(2*(5/3)))
Dphi_fit = k * r_crop**(5/3)

# Scatter plot of r vs autocorrelation
plt.scatter(r_crop, Dphi_avg_crop, s=1)
plt.plot(r_crop, Dphi_fit, color='red')
plt.xlabel("Radial Distance r")
plt.ylabel("Phase Structure Function D$\phi$")
plt.legend()
plt.show()