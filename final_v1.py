import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def pli(image):
    return plt.imshow(image.T, origin="lower")


def load_fits_data(file_path, file_name):
    '''Load FITS data and compute mean across frames.'''

    data = pf.getdata(file_path+file_name)
    mean_data_frames = np.mean(data, axis=0)
    return mean_data_frames

def exp_data_processing(center_c, shifts, phase_screen, D):
#    '''Process experimental data to compute gradient images.'''
    X = int(D)
    Y = int(D)

    grad_imgX = np.zeros((X, Y))
    grad_imgY = np.zeros((X, Y))
    norm = np.zeros((X, Y))

    shift_x_a = shifts[0, 0]
    shift_y_a = shifts[0, 1]
    shift_x_b = shifts[1, 0]
    shift_y_b = shifts[1, 1]
    shift_x_d = shifts[3, 0]
    shift_y_d = shifts[3, 1]

    I_a = phase_screen[center_c[0] + shift_x_a - D//2:center_c[0] + shift_x_a + D//2,
                       center_c[1] + shift_y_a - D//2 :center_c[1] + shift_y_a + D//2]
    I_b = phase_screen[center_c[0] + shift_x_b - D//2:center_c[0] + shift_x_b + D//2,
                       center_c[1] + shift_y_b - D//2 :center_c[1] + shift_y_b + D//2]
    I_c = phase_screen[center_c[0] - D//2:center_c[0] + D//2,
                       center_c[1] - D//2 :center_c[1] + D//2]
    I_d = phase_screen[center_c[0] + shift_x_d - D//2:center_c[0] + shift_x_d + D//2,
                       center_c[1] + shift_y_d - D//2 :center_c[1] + shift_y_d + D//2]
    
    grad_imgX[0:D-1, 0:D-1] =  I_a + I_c - I_b - I_d
    grad_imgY[0:D-1, 0:D-1] =  I_a + I_b - I_c - I_d    

    # For now we use the local normalisation
    normtmp = I_a + I_b + I_c + I_d
    norm[0:D-1, 0:D-1] = normtmp

    grad_imgX = grad_imgX / (norm+1) 
    grad_imgY = grad_imgY / (norm+1)

    return grad_imgX, grad_imgY 

#%%
# Load experimental data
path = "/Users/faure/OneDrive/Bureau/git_kraken/phase-screens/bench_data/"
f_list = ["bg_10000.fits", "no-phase_screen_10000.fits", "phase_screen_10000.fits"]
bg = load_fits_data(path, f_list[0])
phase_screen = load_fits_data(path, f_list[2]) - bg

# Parameter for the function
x = 315 - 94
y = 121 - 100
diameter = np.sqrt(x**2 + y**2)
r = diameter/2 - 30
D = int(2*r)

shifts = np.array([[211 - 210, 374 - 109],   # Shift for A
                   [473 - 210, 374 - 109],   # Shift for B
                   [210 - 210, 109 - 109],   # Shift for C
                   [499 - 210, 109 - 109]])  # Shift for D

center_C = np.array([200, 115])  # Center coordinates of pupil C

# Process experimental data
grad_imgX, grad_imgY = exp_data_processing(center_C, shifts, phase_screen, D)

pli(grad_imgX)
plt.title('Phase Gradient in X')
plt.colorbar()
plt.show()

pli(grad_imgY)
plt.title('Phase Gradient in Y')
plt.colorbar()
plt.show()




#%%
# Save the phase vector gradient 
# saving as .npz file (numpy compressed format)
#np.savez("gradients_xy.npz", gradX=grad_imgX, gradY=grad_imgY)

#%%

'''COMMENTS'''
# ----- Normalisation options -----
#normtmp = I_a + I_b + I_c + I_d
#norm[0:D-1, 0:D-1] = normtmp
#normglobal = np.mean(normtmp)

#normalisation_type = 'local'  # 'local' or 'global'
#if normalisation_type == 'local':
#    grad_imgX = grad_imgX / (norm+1)
#    grad_imgY = grad_imgY / (norm+1)
#else:
#    grad_imgX = grad_imgX / (normglobal)
#    grad_imgY = grad_imgY / (normglobal)

# ----- Normalisation idea with no phase screen -----
# We can use the no phase screen data to compute the normalisation factor but the different intensity levels of the phase and no phase screen must accounted.


# Clovis 28/11

def phase_gradient_vector_cov(phase_derivative_x, phase_derivative_y  ):
    n = np.shape(phase_derivative_x)[0]
    
    # Extended phase screen
    ext_phase_derivative_x = np.zeros((2*n,2*n))
    ext_phase_derivative_x[0:n,0:n] = phase_derivative_x
    ext_phase_derivative_y = np.zeros((2*n,2*n))
    ext_phase_derivative_y[0:n,0:n] = phase_derivative_y

    # Autocorrelations
    auto_dx = (np.fft.ifft2(np.abs(np.fft.fft2(ext_phase_derivative_x))**2)).real
    auto_dy = (np.fft.ifft2(np.abs(np.fft.fft2(ext_phase_derivative_y))**2)).real
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
    
    return covariance
#Applying to experimental data
cov_experimental = phase_gradient_vector_cov(grad_imgX, grad_imgY)
pli(cov_experimental)
plt.title('Covariance from Experimental Data')
plt.colorbar()
plt.show()


# And to theory :
N= np.shape(grad_imgX)[0]  #2048
#ncrop = N #200
x = np.arange(N)-N//2

phase_screen_array_tot = []
no_screens = 50 #50
k2 = np.fft.fftshift(x[None,:]**2 + x[:,None]**2) + 1.0
for i in range(no_screens):
    phase_img = np.fft.fft2(k2**(-11/12) * np.exp(2j*np.pi*np.random.rand(N,N))).real 
    phase_screen_array_tot.append(phase_img[0:N,0:N])

cov = 0.0
for i in range(no_screens):
    phase_screen_array = phase_screen_array_tot[i] #np.mean(phase_screen_array, axis=0) phase_screen_array[0] 
    phase_dx = np.zeros_like(phase_screen_array)
    phase_dy = np.zeros_like(phase_screen_array)

    phase_dx[:-1, :] = phase_screen_array[:-1, :] - phase_screen_array[1:, :]
    phase_dy[:, :-1] = phase_screen_array[:, :-1] - phase_screen_array[:, 1:]


    covariance = phase_gradient_vector_cov(phase_dx, phase_dy)
    cov = cov + covariance
cov_theoric = cov / no_screens

pli(cov_theoric)
plt.title('Covariance from Theory')
plt.colorbar()
plt.show()
print(np.max(cov_theoric))


# Creating r values
n = cov_experimental.shape[0]
tmp = np.arange(n) - n//2
X, Y = np.meshgrid(tmp, tmp, indexing='ij')
r = np.sqrt(X**2 + Y**2)



def fit_linear(r, cov): 
    slope, intercept, _, _, _ = stats.linregress(np.log(r.flatten()), np.log(cov.flatten()))
    y_fit = np.exp(intercept + slope * np.log(r.flatten()))
    return y_fit.reshape(r.shape), slope, intercept

# Extracting the interest region of the covariances
r_limit = np.max(r)
mask = (r > 0) & (r < r_limit)
r_section = r[mask]
cov_interest_theoric = cov_theoric[mask] 
#cov_interest_theoric = cov_interest_theoric + np.max(cov_interest_theoric)*0.25
#cov_interest_experimental = cov_experimental[mask]



#fitting linear regression
y_fit_theoric, slope_theoric, intercept_theoric = fit_linear(r_section, cov_interest_theoric)
#y_fit_experimental, slope_experimental, intercept_experimental = fit_linear(r_section, cov_interest_experimental)



plt.figure()
plt.scatter(r_section.flatten(),  cov_interest_theoric.flatten(), s=1, color='b', label = "Theoric simulated turbulence" )
plt.plot(r_section.flatten(), y_fit_theoric, color='blue', label="Theoric Fit, slope={:.2f}".format(slope_theoric))
#plt.scatter(r_section.flatten(),  cov_interest_experimental.flatten(), s=1, color='r', label = "Experimental data" )  
#plt.plot(r_section.flatten(), y_fit_experimental, color='red', label="Data Fit, slope={:.2f}".format(slope_experimental))
#plt.plot(r.flatten(), 0.5*laplacian_dphi(r.flatten()),  linewidth=0.6, color='r' )  
plt.legend(loc='lower left', fontsize=6) 
plt.loglog()


plt.title('Covariance Interest Region ')
plt.colorbar()  
plt.show()


#Finding the appropriate r : 



#Taking r_limit 5-100 and constant 0.3-0.4

# Define ranges
limit_r_values = range(1, 100)
constant_factor_values = range(-200, 0)

# Create storage matrix
slope_matrix = np.zeros((len(limit_r_values), len(constant_factor_values)))

# Convert ranges to list to allow indexing
limit_r_list = np.linspace(1, 100, 50)
constant_factor_list = np.linspace(-0.02, 0.01, 50)

# 2D loop
for i, limit_r in enumerate(limit_r_list):
    for j, constant_factor in enumerate(constant_factor_list):
        
        mask = (r > 0) & (r < limit_r) 
        r_boucle = r[mask]
        cov_boucle = cov_theoric[mask] + constant_factor

        
        # linear regression
        y_fit, slope, intercept = fit_linear(r_boucle, cov_boucle)
        '''
        if round(slope,2) < -0.32 and round(slope,2) > -0.34:
            plt.scatter(r_boucle.flatten(),  cov_boucle.flatten(), s=1, color='b', label = "Theoric simulated turbulence" )
            plt.plot(r_boucle.flatten(), y_fit, color='blue', label="Theoric Fit, slope={:.2f}".format(slope))
            plt.loglog()
            plt.legend(loc='lower left', fontsize=6)
            plt.show()
        '''
        # store in matrix
        slope_matrix[i, j] = slope
         

mask_valid = (slope_matrix >= -2/3) & (slope_matrix <= 0)
slope_matrix_filtered = np.where(mask_valid, slope_matrix, np.nan)
slope_matrix_filtered = slope_matrix.copy()

# Plot heatmap of k as function of limit_r 
plt.figure(figsize=(10, 6))
plt.imshow(slope_matrix_filtered.T, extent=[min(limit_r_values), max(limit_r_values), min(constant_factor_values), max(constant_factor_values)], aspect='auto', cmap='viridis')

plt.colorbar(label="Slope value")
plt.title("Slope as function of limit_r and constant_factor for restricted values, With a mask to select only slope [0, -2/3], on a 336*336 image")
plt.ylabel("constant k added to covariance")
plt.xlabel("limit_r")
limit_r_list = np.array(limit_r_list)
constant_factor_list = np.array(constant_factor_list)/10000
plt.contour(limit_r_list, constant_factor_list, slope_matrix_filtered.T, levels = [-1/3])
plt.show()

'''
#Ponderation of the covariance : 
r_flat = r.flatten()
cov_flat = cov_theoric.flatten()

# Find all unique radii (float values)
unique_r, inverse_idx = np.unique(r_flat, return_inverse=True)

# Sum covariance for each unique radius
sums = np.bincount(inverse_idx, weights=cov_flat)

# Count number of pixels for each radius
counts = np.bincount(inverse_idx)

# Avoid division by zero
radial_profile = sums / np.maximum(counts, 1)

# Plot
plt.scatter(unique_r, radial_profile)
plt.xscale("log")
plt.yscale("log")
plt.title("Radial average (float radii)")
plt.show()
'''