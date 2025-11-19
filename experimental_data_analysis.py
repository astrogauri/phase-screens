import astropy.io.fits as pf
from cv2 import phase
import numpy as np
import matplotlib.pyplot as plt

def pli(image):
    return plt.imshow(image.T,origin="lower")

# Load experimental data
path = "/Users/gauripadalkar/Desktop/phase-screens/bench_data/"
f_list = ["bg_10000.fits", "no-phase_screen_10000.fits", "phase_screen_10000.fits"]

def mean_frames (file_path):
    data = pf.getdata(file_path)
    mean_data = np.mean(data, axis=0)
    return mean_data


# Background
bg = mean_frames(path + f_list[0])
pli(bg)

# No Phase Screen
no_phase = mean_frames(path + f_list[1]) - bg
pli(no_phase)

# With Phase Screen
phase_screen = mean_frames(path + f_list[2]) - bg
pli(phase_screen)

#%%
# C 
x = 315.4 - 93.8
y = 120.6 - 100.4

D = np.sqrt(x**2 + y**2)
r = D/2 - 20

def circle(x,y,R):
    t = np.linspace(0,2*np.pi,222)
    plt.plot(R*np.cos(t)+x, R*np.sin(t)+y, color='r')

circle (200,115,r)
pli(phase_screen)

nx = 640
ny = 480 
x = np.arange(nx) -200
y = np.arange(ny) -115
xx, yy = np.meshgrid(x,y, indexing='ij')

mask_c = np.sqrt(xx**2 + yy**2) < r

new_c = phase_screen[mask_c]

new_c.shape

just_new_c= np.zeros([nx, ny])
just_new_c[mask_c] = new_c
#pli(just_new_c)

#%%
def meshgrid(nx, ny, center_x, center_y, shift_x, shift_y):
    x = np.arange(nx) - center_x - shift_x
    y = np.arange(ny) - center_y - shift_y
    xx, yy = np.meshgrid(x,y, indexing='ij')
    return xx, yy

#%%
# D
shift_pix_x_d = 499 - 210 
shift_pix_y_d = 109 - 109

xx_d, yy_d = meshgrid(nx, ny, 200, 115, shift_pix_x_d, shift_pix_y_d)
mask_d = np.sqrt(xx_d**2 + yy_d**2) < r

new_d = phase_screen[mask_d]
just_new_d = np.zeros([nx, ny])
just_new_d[mask_d] = new_d
#pli(just_new_d)

# A
shift_pix_x_a = 211.2 - 209.8 
shift_pix_y_a = 373.6 - 109.4

xx_a, yy_a = meshgrid(nx, ny, 200, 115, shift_pix_x_a, shift_pix_y_a)
mask_a = np.sqrt(xx_a**2 + yy_a**2) < r

new_a = phase_screen[mask_a]
just_new_a = np.zeros([nx, ny])
just_new_a[mask_a] = new_a
pli(just_new_a)

# B
shift_pix_x_b = 472.9 - 209.8
shift_pix_y_b = 373.6 - 109.4

xx_b, yy_b = meshgrid(nx, ny, 200, 115, shift_pix_x_b, shift_pix_y_b)
mask_b = np.sqrt(xx_b**2 + yy_b**2) < r

new_b = phase_screen[mask_b]
just_new_b = np.zeros([nx, ny])
just_new_b[mask_b] = new_b
#pli(just_new_b)

