# 1) Removes the data from the sponge layer
# 2) Calculates the reference PV in Fortran 
# 3) Calculates the cyclonic and anticyclonic LWA in Fortran

#Giorgio M. Sarro
#Fortran is embedded in this python code. Make sure that you have a working fortran compiler.
#All output is saved in netCDF
#_____________________________________#
import numpy as np
import sys
from netCDF4 import Dataset
import subprocess
import os
import xarray as xr

#Specify the constants in your model. (See Lutsko and Hell 2021)
C= 2.0
Er= 0.1
U_1 = 0.5
L_arr= 0.15
U_arr = 1

Lstr= str(L)
Cstr= str(C)
Estr= str(Er)
Ustr= str(U_1)


#Specify where the model data is stored
dir_in='/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/N128_%s_%s_%s_%s.3d.nc'%(Lstr,Cstr,Estr,Ustr)
# Specify the name and the location for the saved output
dir_out = '/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/N128_%s_%s_%s_%s.'%(Lstr,Cstr,Estr,Ustr) 
# Specify where the fortran code is stored (rp2.f90 and rp4.f90)
dir_fortran ='2-LayerQG/Local_Wave_Activity/Make_LWA/Calculate_LWA.py'

#_____________________________________#
#The following code opens the PV model data, removes the data in the sponge layer. 
#We assume a monotonic increase in the zonal and time-mean PV, and 1 hemisphere!
#If the sponge layer is not removed, the resulting reference state PV will be inconsistent 

out_name = 'QGPV.nc'
ds = xr.open_dataset(dir_in)
ds_cropped = ds 
q1_mean = np.nanmean(ds_cropped['q1'], axis=2) #Take the zonal mean PV
# Find zonal mean location of maximum and minimum
location_max = np.argmax(q1_mean, axis=1)
location_min = np.argmin(q1_mean, axis=1)

# Find zonal mean value of maximum and minimum
value_max = np.max(q1_mean, axis=1)
value_min = np.min(q1_mean, axis=1)

# Replace values in the sponge layer with the maximum and minimum PV respectively.
for i in range(ds.sizes['time']):
    ds_cropped['q1'][i, location_max[i]:, :] = value_max[i]
    ds_cropped['q1'][i, :location_min[i], :] = value_min[i]

#Save the file with the name that fortran can read
ds_cropped = xr.Dataset({'q1': ds_cropped['q1']})
ds_cropped.to_netcdf(os.path.join(dir_fortran, out_name))
ds.close()
ds_cropped.close()
os.system('chmod 744 QGPV.nc')

#_____________________________________#
#Now we run the first Fortran code to obtain the reference PV
#The details of the following lines will depend on your fortran compiler
#Open the fortran code and modify the details of the x and y grid if necessary!!!

os.system('module load netcdf-fortran')
os.system('gfortran -c -g -fcheck=all $NC_INC rp2.f90 -o rp2.o')
os.system('gfortran -o rp2 rp2.o $NC_LIB')
os.system('./rp2')
        
#_____________________________________#
#Now we run the first Fortran code to calculate cyclonic and anticyclonic LWA
#Open the fortran code and modify the details of the x and y grid if necessary!!!

os.system('gfortran -c -g -fcheck=all $NC_INC rp4.f90 -o rp4.o')
os.system('gfortran -o rp4 rp4.o $NC_LIB')
os.system('./rp4')
        

#_____________________________________#
#remove temporary codes
os.system('rm -f rp2.o')
os.system('rm -f rp4.o')
os.system('rm -f rp2')
os.system('rm -f rp4')
os.system('rm -f QGPV.nc')
#_____________________________________#
#move outputs
os.system('mv %swac1.nc %swac1_2.nc'%(dir_fortran,dir_out))
os.system('mv %swaa1.nc %swaa1_2.nc'%(dir_fortran,dir_out))
os.system('mv %sqref1.nc %sqref1_2.nc'%(dir_fortran,dir_out))

#_____________________________________#
#remove resulting files from the fortran directory
os.system('rm -f wac1.nc')
os.system('rm -f waa1.nc')
os.system('rm -f qref1.nc')

#_____________________________________#

