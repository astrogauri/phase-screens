import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt

def pli(image):
    return plt.imshow(image.T, origin="lower")


def load_fits_data(file_path, file_name):
    '''Load FITS data and compute mean across frames.'''

    data = pf.getdata(file_path+file_name)
    mean_data_frames = np.mean(data, axis=0)
    return mean_data_frames

def exp_data_processing(center_c, shifts, phase_screen, D):
#    '''Process experimental data to compute gradient images.'''
    X = int(2*D)
    Y = int(2*D)

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
path = "/Users/gauripadalkar/Desktop/phase-screens/bench_data/"
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
np.savez("gradients_xy.npz", gradX=grad_imgX, gradY=grad_imgY)

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