import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%%
def pli(image):
    return plt.imshow(image.T,origin="lower")

#%%
data = np.load("gradients_xy.npz")
gradX = data["gradX"]
gradY = data["gradY"]


#%%
def phase_gradient_vector_cov(grad_x,grad_y):
    n = grad_x.shape[0]//2

    # we already have an extended phase screen in grad_x and grad_y

    # Autocorrelations
    auto_dx = (np.fft.ifft2(np.abs(np.fft.fft2(grad_x))**2)).real
    auto_dy = (np.fft.ifft2(np.abs(np.fft.fft2(grad_y))**2)).real

    # Shift zero frequency component to center
    auto_dx = np.fft.fftshift(auto_dx)
    auto_dy = np.fft.fftshift(auto_dy)

    # Crop central region
    auto_dx = auto_dx[n//2:n//2+n, n//2:n//2+n]
    auto_dy = auto_dy[n//2:n//2+n, n//2:n//2+n]

    '''IS THERE A NEED TO NORMALIZE HERE?'''
    # Normalization
    # support = np.zeros((2*n,2*n))
    # support[0:n,0:n] = 1
    support = ((grad_x!=0) | (grad_y!=0)).astype(float) 

    normalization = (np.fft.ifft2(np.abs(np.fft.fft2(support))**2)).real
    normalization = np.fft.fftshift(normalization)
    normalization = normalization[n//2:n//2+n, n//2:n//2+n] 

    # Applying normalization
    auto_dx_normalized = auto_dx / normalization
    auto_dy_normalized = auto_dy / normalization  

    covariance = auto_dx_normalized + auto_dy_normalized

    return grad_x, grad_y, auto_dx_normalized, auto_dy_normalized, covariance

#%%
# Covariance of phase gradient vector
_, _, _, _, covariance = phase_gradient_vector_cov(gradX, gradY)
pli(covariance)
plt.colorbar()
plt.show()

#%%
# Creating r values
n = np.shape(covariance)[0]
tmp = np.arange(n) - n//2
X, Y = np.meshgrid(tmp, tmp, indexing='ij')
r = np.sqrt(X**2 + Y**2)

# Theroretical curve for comparison
def laplacian_dphi(r):
    return (25/9) * r**(-1/3)

# Masking r values 
mask = (r > 0) & (r < 50)
r = r[mask]
covariance = covariance[mask]

# Theoretical(laplacian_dphi) plot with Covariance plot
K =  5.3e-2
plt.scatter(r.flatten(),  covariance.flatten(), s=1, color='b' )  
plt.plot(r.flatten(), K * laplacian_dphi(r.flatten()),  linewidth=0.6, color='r' )  
#plt.loglog()

# Linear regression for fit
slope, intercept, _, _, _ = stats.linregress(np.log(r.flatten()), np.log(covariance.flatten()))
print("Slope from linear regression: ", slope)
print("Intercept from linear regression: ", intercept)

# Plotting the fit with the experimental data
y_fit = np.exp(intercept + slope * np.log(r.flatten()))
plt.figure()
plt.scatter(r.flatten(), covariance.flatten(), s=1, label="Data")
plt.plot(r.flatten(), y_fit, color='red', label="Fit")
plt.xlabel("r")
plt.ylabel("Covariance of phase gradient vector")
plt.loglog()
plt.legend()
