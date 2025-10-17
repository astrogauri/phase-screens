import numpy as np
import matplotlib.pyplot as plt

def pli(image):
    return plt.imshow(image.T,origin="lower")

def phase_structure_function(phase):
    '''Computes the phase structure function from a given simulated phase screen'''

    n = 100
    # Cropping phase screen 
    phase_crop = phase[0:n, 0:n]

    # Computing autocorrelation for the phase screen
    ext_phase_screen = np.zeros((2*n,2*n))
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

    # Computing phase structure function
    # n//2 is the index used to get the center value of autocorrelation
    Dphi = 2*(auto_normalized[n//2,n//2] - auto_normalized) 

    return phase_crop, auto, normalization, auto_normalized,Dphi


# Simulated phase screen for testing
N=200
x = np.arange(N)-N//2
k2 = np.fft.fftshift(x[None,:]**2 + x[:,None]**2) + 1.0
phase_img = np.fft.fft2(k2**(-11/12) * np.exp(2j*np.pi*np.random.rand(N,N))).real 

phase_crop, auto, normalization, auto_normalized, Dphi = phase_structure_function(phase_img)

# Plots

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
plt.title("Phase Structure Function")
plt.colorbar(im4)

plt.tight_layout()
plt.show()
