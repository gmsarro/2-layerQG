### This code computes the eddy growth rate and its mode structure for a 2-layer QG model. ####
#Computes the fastest and mean eddy growth rates in a 2-layer QG model.
#This is a modified version of the Phillips model.
#Written by Giorgio M. Sarro and Noboru Nakamura

import numpy as np
from scipy.linalg import eig
import netCDF4 as nc
from build_matrices import *


beta= 0.2
resolution = 600 #resolution of the wavenumber space
max_wavenumber = 3 #highest wavenumber
range_k = np.linspace(0, max_wavenumber, resolution) #range of the wavenumber space 
sponge_layer_min = 11 #boundary of the bottom sponge layer (where eddies cannot grow)
sponge_layer_max = -12 #boundary of the top sponge layer

#Datapath to your model.
data_path = "/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/N128_0.0_2.0_0.1_1.0.3d.nc"
#Output save name
output_file = "new_eigenvalue_results_L0.0_U1.0.nc"

#==========================================================#
# Load your 3D model data
# Load data: take the time and zonal mean of the upper (u1) and lower (u2) winds
with nc.Dataset(data_path, 'r') as nc_file:
    y = nc_file.variables['y'][:]
    u1_mean = np.mean(np.mean(nc_file.variables['u1'][:], axis=2), axis=0)
    u2_mean = np.mean(np.mean(nc_file.variables['u2'][:], axis=2), axis=0)
    
#Remove the Sponge Layer of the mode (meridional boundaries)
u1 = np.copy(u1_mean)*0
u2 = np.copy(u2_mean)*0
u1[sponge_layer_min:sponge_layer_max] = np.copy(u1_mean[sponge_layer_min:sponge_layer_max])
u2[sponge_layer_min:sponge_layer_max] = np.copy(u2_mean[sponge_layer_min:sponge_layer_max])

dy = y[1] - y[0]  # Assuming uniform spacing in y
#dy can also be written as L_d/len(y); where L_d is the Rossby radius of deformation, and len(y) is the length of the meridional domain.
n_2= int(len(y)*2)
n= int(len(y))
half_maxtrix = n-2
growth = np.zeros(resolution)
mean_growth = np.zeros(resolution)
kk = np.zeros(resolution)
loc= 0


#==========================================================#
#We loop through the wavenumbers to find the growth at each one
for rk in range_k:
    M, N = build_matrices(u1, u2, beta, dy, n_2, rk, half_maxtrix,n)
    evals, V = eig(M, N)
    gr = evals.imag*rk
    growth[loc] = np.max(gr)
    mean_growth[loc] = np.mean(np.abs(gr))  
    kk[loc] = rk
    print('wavenumber ',rk,' of ', max_wavenumber)
    loc += 1
    
#==========================================================#
#This part of the code extracts the mode structure of the wave at the peak growth rate
peak_index = np.argmax(growth)
rk = kk[peak_index] #wavenumber of peak growth 
M, N = build_matrices(u1, u2, beta, dy, n_2, rk, half_maxtrix,n)
evals, V = eig(M, N)
peak_mode_index = np.argmax(evals.imag * rk)
peak_mode_structure = V[:, peak_mode_index]
peak_mode_structure_upper_img = np.copy(y)*0
peak_mode_structure_lower_img = np.copy(y)*0
peak_mode_structure_upper_real = np.copy(y)*0
peak_mode_structure_lower_real = np.copy(y)*0

#Here we extract the mode, real and imaginary, for the upper and lower layers of the model. 
peak_mode_structure_upper_img[1:-1] = peak_mode_structure.imag[:half_maxtrix]
peak_mode_structure_lower_img[1:-1] = peak_mode_structure.imag[half_maxtrix:]
peak_mode_structure_upper_real[1:-1] = peak_mode_structure.real[:half_maxtrix]
peak_mode_structure_lower_real[1:-1] = peak_mode_structure.real[half_maxtrix:]

#==========================================================#
#Save the output as a NetCDF file
with nc.Dataset(output_file, 'w', format='NETCDF4_CLASSIC') as nc_file:
    # Create dimensions for k_values
    nc_file.createDimension('k_dim', len(kk))
    # Create dimension for 'y'
    nc_file.createDimension('y_dim', len(y))
    # Create variables to store the data
    nc_file.createVariable('k', 'f8', ('k_dim',))
    nc_file.createVariable('y', 'f8', ('y_dim',))
    nc_file.createVariable('largest_imaginary_eigenvalues', 'f8', ('k_dim',))
    nc_file.createVariable('mean_imaginary_eigenvalues', 'f8', ('k_dim',))
    nc_file.createVariable('optimal_mode_upper_img', 'f8', ('y_dim', ))
    nc_file.createVariable('optimal_mode_lower_img', 'f8', ('y_dim', ))
    nc_file.createVariable('optimal_mode_upper_real', 'f8', ('y_dim', ))
    nc_file.createVariable('optimal_mode_lower_real', 'f8', ('y_dim', ))

    # Save data into variables
    nc_file['k'][:] = kk
    nc_file['y'][:] = y
    nc_file['largest_imaginary_eigenvalues'][:] = growth
    nc_file['mean_imaginary_eigenvalues'][:] = mean_growth
    nc_file['optimal_mode_upper_img'][:] = peak_mode_structure_upper_img
    nc_file['optimal_mode_lower_img'][:] = peak_mode_structure_lower_img
    nc_file['optimal_mode_upper_real'][:] = peak_mode_structure_upper_real
    nc_file['optimal_mode_lower_real'][:] = peak_mode_structure_lower_real
