#written by Noboru Nakamura and G.M. Sarro

import numpy as np
from scipy.linalg import eig

# Periodic Laplacian operator
def laplacian(f, dx):
    return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2
def gradient(f, dx):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
def moist_matrix(L,U1,U2):
	# Calculate the fastest growing modes by incorporating moisture
    #Computes the upper PV mode and precipitation at the fastest growth
    delta = 1.0  # lower layer thickness (1: equal thickness)
    el = 0.
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
        aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+U2*E/delta
        aa[0,1] = U1/(2.-delta)-U2*E/delta+E*(D+C)/delta
        aa[0,2] = U2*E
        bb[0,0] = -(rk*rk+el*el+1./(2.-delta)) + E/delta
        bb[0,1] = 1./(2.-delta)-E/delta
        bb[0,2] = E
        aa[1,0] = (U2/delta)*(1.-E)
        aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+U2*E/delta-E*(D+C)/delta+(0.+1.j)*R*(rk*rk+el*el)/rk
        aa[1,2] = -U2*E
        bb[1,0] = (1.-E)/delta
        bb[1,1] = -(rk*rk+el*el+1./delta) + E/delta
        bb[1,2] = -E
        aa[2,0] = (U2/delta)/(1.-L)
        aa[2,1] = -U2*(rk*rk+el*el)-(U2/delta)/(1.-L)+G+(0.+1.j)*R*(rk*rk+el*el)/rk
        aa[2,2] = U2*L/(1.-L)
        bb[2,0] = 1./(delta*(1.-L))
        bb[2,1] = -(rk*rk+el*el+1./(delta*(1.-L)))
        bb[2,2] = L/(1.-L)
        evals, V = eig(aa, bb)
        gr = evals.imag*rk
        growth[m] = np.max(gr)
        kk[m] = rk
        m = m+1

    peak_index = np.argmax(growth)
    rk = kk[peak_index] #wavenumber of peak growth 
    aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+U2*E/delta
    aa[0,1] = U1/(2.-delta)-U2*E/delta+E*(D+C)/delta
    aa[0,2] = U2*E
    bb[0,0] = -(rk*rk+el*el+1./(2.-delta)) + E/delta
    bb[0,1] = 1./(2.-delta)-E/delta
    bb[0,2] = E
    aa[1,0] = (U2/delta)*(1.-E)
    aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+U2*E/delta-E*(D+C)/delta+(0.+1.j)*R*(rk*rk+el*el)/rk
    aa[1,2] = -U2*E
    bb[1,0] = (1.-E)/delta
    bb[1,1] = -(rk*rk+el*el+1./delta) + E/delta
    bb[1,2] = -E
    aa[2,0] = (U2/delta)/(1.-L)
    aa[2,1] = -U2*(rk*rk+el*el)-(U2/delta)/(1.-L)+G+(0.+1.j)*R*(rk*rk+el*el)/rk
    aa[2,2] = U2*L/(1.-L)
    bb[2,0] = 1./(delta*(1.-L))
    bb[2,1] = -(rk*rk+el*el+1./(delta*(1.-L)))
    bb[2,2] = L/(1.-L)
    evals, V = eig(aa, bb)
    gr = evals.imag*rk

    peak_mode_index = np.argmax(evals.imag * rk)

    
    # Extract eigenvector components for psi_1' and psi_2'
    psi_1_prime = V[0, peak_mode_index]  # Upper layer streamfunction
    psi_2_prime = V[1, peak_mode_index]  # Lower layer streamfunction
    m_prime = V[2, peak_mode_index]  # Lower layer m

    x= np.arange(0,6*np.pi,.1)
    dx = .1
    psi_1 = (np.outer(psi_1_prime.real, np.cos(rk * x)) - np.outer(psi_1_prime.imag, np.sin(rk * x)))[0,:]
    psi_2 = (np.outer(psi_2_prime.real, np.cos(rk * x)) - np.outer(psi_2_prime.imag, np.sin(rk * x)))[0,:]
    m_var = (np.outer(m_prime.real, np.cos(rk * x)) - np.outer(m_prime.imag, np.sin(rk * x)))[0,:]
    
    # Calculate QGPV for the upper layer (q'_1)
    q_1_prime = laplacian(psi_1, dx) - (psi_1 - psi_2)
    
    # Calculate diabatic source (P') using eq. (10) from the PDF
    # Compute P 
    dPsi1_dx = gradient(psi_1, dx)
    dPsi2_dx = gradient(psi_2, dx)
    dm_dx = gradient(m_var, dx)

    # First term of the equation for P'
    P_term1 = ( U2 * dPsi1_dx) - ( U2 * dPsi2_dx) + (U2 * dm_dx)
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
        
    return kk,growth,q_1_prime,P