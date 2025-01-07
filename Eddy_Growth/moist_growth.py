#Author: Giorgio Sarro (Uchicago)

### This code simulates growing disturbances following the equations in Lapeyre and Held (2004) ###
## Precipitation is solved using the moist PV equation, assuming *near saturation*


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Set up parameters
Lx = 72. #size of x 
Nx = 128  # Number of grid points
dx = Lx / Nx
dt = 0.025  # Time step
total_time = 40  # Total simulation time
num_steps = int(total_time / dt)

# Parameters for the problem
U1, U2 = 1.0, 0.4  # Example values
beta=0.2 # nondimensional beta
Ld=1 # deformation radius
C = 2. #linearized Clausius-Clapeyron parameter

# Define the range for L and tau
L_values = np.arange(0.0, 0.55, 0.05)
tau_values = np.arange(10, 31, 1)


###############################################################

# Function to save output to a netCDF file
def save_to_netcdf(filename, L, tau, max_growth, x, Psi_1_at_max_growth, Psi_2_at_max_growth, time_array, amplitudes_Psi1_smoothed, amplitudes_Psi2_smoothed, P):
    with Dataset(filename, 'w', format='NETCDF4') as dataset:
        # Create dimensions
        x_dim = dataset.createDimension('x', len(x))
        time_dim = dataset.createDimension('time', len(time_array))

        # Create variables
        L_var = dataset.createVariable('L', 'f4')
        tau_var = dataset.createVariable('tau', 'f4')
        max_growth_var = dataset.createVariable('max_growth', 'f4')
        x_var = dataset.createVariable('x', 'f4', ('x',))
        Psi_1_var = dataset.createVariable('Psi_1_at_max_growth', 'f4', ('x',))
        Psi_2_var = dataset.createVariable('Psi_2_at_max_growth', 'f4', ('x',))
        P_var = dataset.createVariable('P_at_max_growth', 'f4', ('x',))
        time_var = dataset.createVariable('time', 'f4', ('time',))
        Psi1_smooth_var = dataset.createVariable('amplitudes_Psi1_smoothed', 'f4', ('time',))
        Psi2_smooth_var = dataset.createVariable('amplitudes_Psi2_smoothed', 'f4', ('time',))

        # Store the data
        L_var[:] = L
        tau_var[:] = tau
        max_growth_var[:] = max_growth
        x_var[:] = x
        Psi_1_var[:] = Psi_1_at_max_growth
        Psi_2_var[:] = Psi_2_at_max_growth
        P_var[:] = P
        time_var[:] = time_array
        Psi1_smooth_var[:] = amplitudes_Psi1_smoothed
        Psi2_smooth_var[:] = amplitudes_Psi2_smoothed

def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='same')

# Periodic Laplacian operator
def laplacian(f, dx):
    return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2

# Periodic gradient operator
def gradient(f, dx):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

# Time-stepping using a simple Euler method for now
def time_step(Psi_1, Psi_2, dx, dt, Ld, m, P):
    # Compute q1, q2, qm
    q1 = laplacian(Psi_1, dx) - (Psi_1 - Psi_2)/Ld**2
    q2 = laplacian(Psi_2, dx) + (Psi_1 - Psi_2)/Ld**2
    qm = laplacian(Psi_2, dx) + (1/(1-L)) * (Psi_1 - Psi_2 + L*m)/Ld**2

    m_old = np.copy(m)

    m = qm-q2


    # Compute the time derivatives using the system of equations
    dq1_dt = -U1 * gradient(q1, dx) - (beta + U1-U2) * gradient(Psi_1, dx) - L * P
    dq2_dt = -U2 * gradient(q2, dx) - (beta - (U1-U2)) * gradient(Psi_2, dx) + L * P - laplacian(Psi_2, dx) / tau

    # Update q_1 and q_2 using Euler method
    q_1_new = q1 + dq1_dt * dt
    q_2_new = q2 + dq2_dt * dt


    # solve for Streamfunction using SOR method
    Psi_1_new=np.copy(Psi_1[:]) # use Psi_1 as the initial guess. This would make convergence faster.
    Psi_2_new=np.copy(Psi_2[:]) # use Psi_1 as the initial guess. This would make convergence faster.

    AC=np.array([1/dx**2,-2/dx**2,1/dx**2]) # array for second-derivative

    # below process solves 
    #$d^2 Psi_1/ d x^2 - (Psi_1-Psi2)/Ld^2 - q1 =0.
    #$d^2 Psi_2/ d x^2 + (Psi_1-Psi2)/Ld^2 - q2 =0.

    # initial values
    nIT=0 #number of iteration
    err=1E+5 #initial error
    # parameters for numerical solver
    maxerr=1E-6 # maximum tolerated error
    maxIT=10000 # maximum number of iteration
    relax=1.5 # relaxation parameter

    # start iteration
    while nIT < maxIT and err > maxerr:
            Psi_1_temp=np.copy(Psi_1_new) # temporary array upate with previous solution
            Psi_2_temp=np.copy(Psi_2_new) # temporary array upate with previous solution
        
            x=Nx-1
            RS=(AC[0]*Psi_1_new[x-1]+AC[1]*Psi_1_new[x]+AC[2]*Psi_1_new[0])-q_1_new[x]-(Psi_1_new[x]-Psi_2_new[x])/Ld**2 
            RP=(AC[0]*Psi_2_new[x-1]+AC[1]*Psi_2_new[x]+AC[2]*Psi_2_new[0])-q_2_new[x]+(Psi_1_new[x]-Psi_2_new[x])/Ld**2 
            Psi_1_new[x]=Psi_1_new[x]-relax*RS/(AC[1]-1/Ld**2)
            Psi_2_new[x]=Psi_2_new[x]-relax*RP/(AC[1]-1/Ld**2)
            for x in range(0,Nx-1):
                RS=(AC[0]*Psi_1_new[x-1]+AC[1]*Psi_1_new[x]+AC[2]*Psi_1_new[x+1])-q_1_new[x]-(Psi_1_new[x]-Psi_2_new[x])/Ld**2 
                RP=(AC[0]*Psi_2_new[x-1]+AC[1]*Psi_2_new[x]+AC[2]*Psi_2_new[x+1])-q_2_new[x]+(Psi_1_new[x]-Psi_2_new[x])/Ld**2 
                Psi_1_new[x]=Psi_1_new[x]-relax*RS/(AC[1]-1/Ld**2)
                Psi_2_new[x]=Psi_2_new[x]-relax*RP/(AC[1]-1/Ld**2)
            err_1=np.max(np.abs(Psi_1_temp[:]-Psi_1_new[:])) 
            err_2 = np.max(np.abs(Psi_2_temp[:]-Psi_2_new[:]))
            err = max(err_1, err_2)
            nIT+=1

    if nIT == maxIT:	
        print('Not fully converged')
#    else:	print('converged at %i th iteration'%nIT)
    
    # Compute P 
    dPsi1_dt = (Psi_1_new-Psi_1)/dt
    dPsi2_dt = (Psi_2_new-Psi_2)/dt
    dm_dt = (m-m_old)/dt

    dPsi1_dx = gradient(Psi_1, dx)
    dPsi2_dx = gradient(Psi_2, dx)
    dm_dx = gradient(m, dx)

    # First term of the equation for P'
    P_term1 = (dPsi1_dt + U2 * dPsi1_dx) - (dPsi2_dt + U2 * dPsi2_dx) + (dm_dt + U2 * dm_dx)
    P_term1 = P_term1 / (1 - L)

    # Second term of the equation for P'
    P_term2 = ((C+ U1-U2) * dPsi2_dx) / (1 - L)

    # Final P' equation
    P = P_term1 + P_term2
    #Remove negative rain
    no_rain_loc = P<0
    P[no_rain_loc] =0 
    #Remove average rain to have net 0 latent heating
    mean_P = np.mean(P)
    P -= mean_P

    
    return Psi_1_new, Psi_2_new , nIT , q1, q_1_new, q_2_new, m, P

# Main loop to run simulations for different L and tau values
for tau in tau_values:
    for L in L_values:
        L= np.round(L,2)
        tau = np.round(tau,2)
        x = np.linspace(0, Lx, Nx, endpoint=False)
        print(f"L={L}, tau={tau}")

        # Set up random initial conditions
        Psi_1 = np.random.uniform(-.1, .1, Nx)
        Psi_2 = np.random.uniform(-.1, .1, Nx)
        # Initialize arrays to store the growth rates
        amplitudes_Psi1 = []
        amplitudes_Psi2 = []
        Psi_1_over_time = []
        Psi_2_over_time = []
        P_over_time = []
        P=0
        m=0

        # Time integration loop
        for step in range(num_steps):
#            print('numner of step: ', step)
            Psi_1, Psi_2, nIT , q1, q_1_new, q_2_new, m, P= time_step(Psi_1, Psi_2, dx, dt, Ld, m, P)
            if nIT >= 10000:
                # Fill remaining arrays with the current values and break the loop
                for _ in range(step, num_steps):
                    amplitudes_Psi1.append(amplitudes_Psi1[-1] if amplitudes_Psi1 else np.sqrt(np.mean(Psi_1**2)))
                    amplitudes_Psi2.append(amplitudes_Psi2[-1] if amplitudes_Psi2 else np.sqrt(np.mean(Psi_2**2)))
                    Psi_1_over_time.append(np.copy(Psi_1))
                    Psi_2_over_time.append(np.copy(Psi_2))
                    P_over_time.append(np.copy(P))
                print('integration broken')
                break
            # Calculate the amplitude (energy) at each time step
            amplitude_Psi1 = np.sqrt(np.mean(Psi_1**2))
            amplitude_Psi2 = np.sqrt(np.mean(Psi_2**2))
    
            amplitudes_Psi1.append(amplitude_Psi1)
            amplitudes_Psi2.append(amplitude_Psi2)
            # Store Psi_1 and Psi_2 at each step
            Psi_1_over_time.append(np.copy(Psi_1))
            Psi_2_over_time.append(np.copy(Psi_2))
            P_over_time.append(np.copy(P))
    
        # Convert to numpy arrays for easier manipulation
        Psi_1_over_time = np.array(Psi_1_over_time)
        Psi_2_over_time = np.array(Psi_2_over_time)
        P_over_time = np.array(P_over_time)

        time_array = np.linspace(0, total_time, num_steps)
        # Calculate the rate of change of amplitude
        dA_dt = np.gradient(amplitudes_Psi1, time_array)

        # Smooth dA_dt and amplitudes_Psi1 using a window of 50
        window_size = 50
        dA_dt_smoothed = moving_average(dA_dt, window_size)
        amplitudes_Psi1_smoothed = moving_average(amplitudes_Psi1, window_size)
        amplitudes_Psi2_smoothed = moving_average(amplitudes_Psi2, window_size)

        # Compute the smoothed growth rate
        growth_rate_smoothed = dA_dt_smoothed / amplitudes_Psi1_smoothed

        # Find the index of the maximum growth rate after the first 100 values
        max_growth_index = np.argmax(growth_rate_smoothed[1:])
        max_growth = np.max(growth_rate_smoothed[1:])


        # Retrieve the Psi_1 and Psi_2 at the time of the fastest growth rate
        Psi_1_at_max_growth = Psi_1_over_time[max_growth_index+1]
        Psi_2_at_max_growth = Psi_2_over_time[max_growth_index+1]
        P_at_max_growth = P_over_time[max_growth_index+1]

        x = np.linspace(0, Lx, Nx, endpoint=False)
        
        # Save results to a netCDF file
        filename = f"/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/Moist_growth/results/growth_L_{L}_tau_{tau}.nc"
        save_to_netcdf(filename, L, tau, max_growth, x, Psi_1_at_max_growth, Psi_2_at_max_growth, time_array, amplitudes_Psi1_smoothed, amplitudes_Psi2_smoothed, P_at_max_growth)

        print(f"Saved results for L={L}, tau={tau} to {filename} with growth= {max_growth}")