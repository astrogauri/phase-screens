import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
def pli(image):
    return plt.imshow(image.T,origin="lower")

# %%
'''Instead of computing the covariance of phase derivative, we define the phase gradient vector with the two components and get the covariance of that. 
   This allows us to plot the covarince (experimental values) as a function of r'''

def phase_gradient_vector_cov(phase):
    n = 100
    
    # Gradient components
    phase_dx = phase[0:n, 0:n] - phase[1:n+1, 0:n] # dphi/dx
    phase_dy = phase[0:n, 0:n] - phase[0:n, 1:n+1] # dphi/dy

    # Extended phase screen
    ext_phase_dx = np.zeros((2*n,2*n))
    ext_phase_dy = np.zeros((2*n,2*n))
    ext_phase_dx[0:n,0:n] = phase_dx
    ext_phase_dy[0:n,0:n] = phase_dy

    # Autocorrelations
    auto_dx = (np.fft.ifft2(np.abs(np.fft.fft2(ext_phase_dx))**2)).real
    auto_dy = (np.fft.ifft2(np.abs(np.fft.fft2(ext_phase_dy))**2)).real
    # Shift zero frequency component to center
    auto_dx = np.fft.fftshift(auto_dx)
    auto_dy = np.fft.fftshift(auto_dy)
    # Crop central region
    auto_dx = auto_dx[n//2:n//2+n, n//2:n//2+n]
    auto_dy = auto_dy[n//2:n//2+n, n//2:n//2+n]

    # Normalization
    support = np.zeros((2*n,2*n))
    support[0:n,0:n] = 1
    normalization = (np.fft.ifft2(np.abs(np.fft.fft2(support))**2)).real
    normalization = np.fft.fftshift(normalization)
    normalization = normalization[n//2:n//2+n, n//2:n//2+n] 

    # Applying normalization
    auto_dx_normalized = auto_dx / normalization
    auto_dy_normalized = auto_dy / normalization  

    covariance = auto_dx_normalized + auto_dy_normalized

    return phase_dx, phase_dy, auto_dx_normalized, auto_dy_normalized, covariance

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

cov = 0.0
for i in range(no_screens):
    _, _, _, _, covariance = phase_gradient_vector_cov(phase_screen_array[i])
    cov = cov + covariance
cov = cov / no_screens

#%%
pli(cov)


n = 100
tmp = np.arange(n) - n//2
X, Y = np.meshgrid(tmp, tmp, indexing='ij')
r = np.sqrt(X**2 + Y**2)

plt.figure()
plt.scatter(r.flatten(), cov.flatten(), s=1)
plt.xlabel("r")
plt.ylabel("Covariance of phase gradient vector")
plt.show()

# %%

lin_reg = LinearRegression.fit(np.log(cov.flatten()),  np.log(r.flatten()))
print("Slope from linear regression (log-log scale): ", lin_reg.coef_)

# Theoretical slope for Kolmogorov turbulence is -5/3
theoretical_slope = -5/3
print("Theoretical slope: ", theoretical_slope)

