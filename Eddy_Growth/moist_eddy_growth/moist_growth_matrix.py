#written by Noboru Nakamura and G.M. Sarro
# Calculate the fastest growing modes by incorporating moisture
#Computes the upper PV mode and precipitation at the fastest growth

import numpy as np
from scipy.linalg import eig

# Periodic Laplacian operator
def laplacian(f, dx):
    return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2
def gradient(f, dx):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
def moist_matrix(L,U1,U2):
    delta = 1.0  # lower layer thickness (1: equal thickness)
    el = 0.
    ep = 0.5
    dk = 4./400.
    D = U1-U2
    B = 0.2
    C = 2.
    R = 0.0
    G = B - (D+C*L)/(delta*(1.-L))
    E = 0.5*L/(1.-L)

    aa = np.zeros((3,3),dtype=complex)
    bb = np.zeros((3,3),dtype=complex)
    growth = np.zeros(400)
    kk = np.zeros(400)

    m = 1
    while(m < 400):
        rk = dk*m   # nondimensional wavenumber
        aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+ep*(0.+1.j)*L*C/rk
        aa[0,1] = U1/(2.-delta)-ep*(0.+1.j)*L*C/rk
        aa[0,2] = -ep*(0.+1.j)*L/rk
        bb[0,0] = -(rk*rk+el*el+1./(2.-delta))
        bb[0,1] = 1./(2.-delta)
        bb[0,2] = 0.
        aa[1,0] = U2/delta-ep*(0.+1.j)*L*C/rk
        aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+(0.+1.j)*R*(rk*rk+el*el)/rk+ep*(0.+1.j)*L*C/rk
        aa[1,2] = ep*(0.+1.j)*L/rk
        bb[1,0] = 1./delta
        bb[1,1] = -(rk*rk+el*el+1./delta)
        bb[1,2] = 0.
        aa[2,0] = U2+(0.+1.j)*(1.-L)*C*ep/rk
        aa[2,1] = -U2-(C+D)-(0.+1.j)*(1.-L)*C*ep/rk
        aa[2,2] = U2-(0.+1.j)*(1.-L)*ep/rk
        bb[2,0] = 1.
        bb[2,1] = -1.
        bb[2,2] = 1.
    
        evals, V = eig(aa, bb)
        gr = evals.imag*rk
        growth[m] = np.max(gr)
        kk[m] = rk
        m = m+1

    peak_index = np.argmax(growth)
    rk = kk[peak_index] #wavenumber of peak growth 
    aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+ep*(0.+1.j)*L*C/rk
    aa[0,1] = U1/(2.-delta)-ep*(0.+1.j)*L*C/rk
    aa[0,2] = -ep*(0.+1.j)*L/rk
    bb[0,0] = -(rk*rk+el*el+1./(2.-delta))
    bb[0,1] = 1./(2.-delta)
    bb[0,2] = 0.
    aa[1,0] = U2/delta-ep*(0.+1.j)*L*C/rk
    aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+(0.+1.j)*R*(rk*rk+el*el)/rk+ep*(0.+1.j)*L*C/rk
    aa[1,2] = ep*(0.+1.j)*L/rk
    bb[1,0] = 1./delta
    bb[1,1] = -(rk*rk+el*el+1./delta)
    bb[1,2] = 0.
    aa[2,0] = U2+(0.+1.j)*(1.-L)*C*ep/rk
    aa[2,1] = -U2-(C+D)-(0.+1.j)*(1.-L)*C*ep/rk
    aa[2,2] = U2-(0.+1.j)*(1.-L)*ep/rk
    bb[2,0] = 1.
    bb[2,1] = -1.
    bb[2,2] = 1.
    evals, V = eig(aa, bb)
    gr = evals.imag*rk
    peak_mode_index = np.argmax(evals.imag * rk)
    
    # Extract eigenvector components
    psi_1_prime = V[0, peak_mode_index]  # Upper layer streamfunction
    psi_2_prime = V[1, peak_mode_index]  # Lower layer streamfunction
    P_prime = 0.5*V[2, peak_mode_index]*L  # Precipitation

    x= np.arange(0,6*np.pi,.1)
    dx = .1
    psi_1 = (np.outer(psi_1_prime.real, np.cos(rk * x)) - np.outer(psi_1_prime.imag, np.sin(rk * x)))[0,:]
    psi_2 = (np.outer(psi_2_prime.real, np.cos(rk * x)) - np.outer(psi_2_prime.imag, np.sin(rk * x)))[0,:]
    P = (np.outer(P_prime.real, np.cos(rk * x)) - np.outer(P_prime.imag, np.sin(rk * x)))[0,:]
    
    # Calculate QGPV for the upper layer (q'_1)
    q_1_prime = laplacian(psi_1, dx) - (psi_1 - psi_2)
    # Calculate QGPV for the upper layer (q'_2)
    q_2_prime = laplacian(psi_2, dx) + (psi_1 - psi_2)
        
    return kk,growth,q_1_prime,q_2_prime, P