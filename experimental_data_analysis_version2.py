import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt

def pli(image):
    return plt.imshow(image.T, origin="lower")

# Load experimental data
path = "/Users/faure/OneDrive/Bureau/git_kraken/phase-screens/bench_data/"
f_list = ["bg_10000.fits", "no-phase_screen_10000.fits", "phase_screen_10000.fits"]

def mean_frames(file_path):
    data = pf.getdata(file_path)
    return np.mean(data, axis=0)

# Background, only 'parasite' light sources
bg = mean_frames(path + f_list[0])

# No Phase Screen, bg + source, no screen phase
no_phase = mean_frames(path + f_list[1]) - bg

# With Phase Screen, bg + source, with phase screen
phase_screen = mean_frames(path + f_list[2]) - bg

# Coordinates to determine radius of C circle
x = 315.4 - 93.8
y = 120.6 - 100.4

D = np.sqrt(x**2 + y**2)
r = D/2 - 20 # We reduce r to get interesting signal only 

nx, ny = 640, 480 #Image size
x0, y0 = 200, 115 #Center of the considered circle



def meshgrid(nx, ny, center_x, center_y, shift_x, shift_y):
    x = np.arange(nx) - center_x - shift_x
    y = np.arange(ny) - center_y - shift_y
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return xx, yy

# Recenters the image for final gradient calculation
def recenter(img, old_cx, old_cy, new_cx, new_cy):
    shift_x = int(new_cx - old_cx)
    shift_y = int(new_cy - old_cy)
    return np.roll(np.roll(img, shift_x, axis=0), shift_y, axis=1)

#Extracting the four pupillas 
# C
xx_c, yy_c = meshgrid(nx, ny, x0, y0, 0, 0)
mask_c = np.sqrt(xx_c**2 + yy_c**2) < r

#just_new_c = np.zeros((nx, ny))
just_new_c = np.ones((nx, ny))
just_new_c[mask_c] = phase_screen[mask_c]

# D
shift_pix_x_d = 499 - 210
shift_pix_y_d = 109 - 109

xx_d, yy_d = meshgrid(nx, ny, x0, y0, shift_pix_x_d, shift_pix_y_d)
mask_d = np.sqrt(xx_d**2 + yy_d**2) < r

just_new_d = np.ones((nx, ny))
just_new_d[mask_d] = phase_screen[mask_d]

# A
shift_pix_x_a = 211.2 - 209.8
shift_pix_y_a = 373.6 - 109.4

xx_a, yy_a = meshgrid(nx, ny, x0, y0, shift_pix_x_a, shift_pix_y_a)
mask_a = np.sqrt(xx_a**2 + yy_a**2) < r

just_new_a = np.ones((nx, ny))
just_new_a[mask_a] = phase_screen[mask_a]


# B
shift_pix_x_b = 472.9 - 209.8
shift_pix_y_b = 373.6 - 109.4

xx_b, yy_b = meshgrid(nx, ny, x0, y0, shift_pix_x_b, shift_pix_y_b)
mask_b = np.sqrt(xx_b**2 + yy_b**2) < r

just_new_b = np.ones((nx, ny))
just_new_b[mask_b] = phase_screen[mask_b]

# Recentering all pupilla center coordinate to be able to calculate gradients
center_ref_x = x0         
center_ref_y = y0

center_c_x, center_c_y = x0, y0
center_d_x = x0 + shift_pix_x_d
center_d_y = y0 + shift_pix_y_d
center_a_x = x0 + shift_pix_x_a
center_a_y = y0 + shift_pix_y_a
center_b_x = x0 + shift_pix_x_b
center_b_y = y0 + shift_pix_y_b

centered_c = recenter(just_new_c, center_c_x, center_c_y, center_ref_x, center_ref_y)
centered_d = recenter(just_new_d, center_d_x, center_d_y, center_ref_x, center_ref_y)
centered_a = recenter(just_new_a, center_a_x, center_a_y, center_ref_x, center_ref_y)
centered_b = recenter(just_new_b, center_b_x, center_b_y, center_ref_x, center_ref_y)

gradx = centered_a + centered_c - (centered_b + centered_d)
grady = centered_a + centered_b - (centered_c + centered_d)

normalization = centered_a + centered_b + centered_c + centered_d

gradx = gradx / normalization
grady = grady / normalization

pli(normalization)
plt.show()

pli(gradx)
plt.title('Grad x phase')
plt.show()

pli(grady)
plt.title('Grad y phase')
plt.show()



#Computing the autocorrelation of gradx and grad y :

def laplacian_dphi(r):
    return (25/9) * r**(-1/3)

def phase_gradient_vector_cov(gradx, grady):
    n = 100
    
    # Gradient components
    phase_dx =gradx # dphi/dx
    phase_dy = grady # dphi/dy

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

