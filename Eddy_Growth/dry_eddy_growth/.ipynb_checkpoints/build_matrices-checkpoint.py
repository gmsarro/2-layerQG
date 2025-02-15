#Authors: Noboru Nakamura and Giorgio M. Sarro

#Build the matrix that solves the linearized PV equation for a perturbation in the form of perturbations in the form A2(y)e^(i(kx-ct)) and A1(y)e^(i(kx-ct))
import numpy as np
def build_matrices(u1, u2, beta, dy, n_2, rk, half_maxtrix, n):
    M = np.zeros((n_2-4,n_2-4))
    N = np.zeros((n_2-4,n_2-4))
    
    for j in range(n-4):
        M[j,j] = -u1[j+1]*(rk**2*dy**2+2.+dy**2)
        M[j,j] += (beta*dy**2-(u1[j+2]+u1[j]-2.*u1[j+1]))
        M[j,j] += (u1[j+1]-u2[j+1])*dy**2
        M[j+half_maxtrix,j+half_maxtrix] = -u2[j+1]*(rk**2*dy**2+2.+dy**2)
        M[j+half_maxtrix,j+half_maxtrix] += (beta*dy**2-(u2[j+2]+u2[j]-2.*u2[j+1]))
        M[j+half_maxtrix,j+half_maxtrix] += -(u1[j+1]-u2[j+1])*dy**2
        M[j,j+half_maxtrix]=u1[j+1]*dy**2
        M[j+half_maxtrix,j]=u2[j+1]*dy**2
        M[j,j+1] = u1[j+1]
        M[j+1,j] = u1[j+2]
        M[j+half_maxtrix,j+half_maxtrix+1] = u2[j+1]
        M[j+half_maxtrix+1,j+half_maxtrix] = u2[j+2]
        
        N[j,j+1] = 1.
        N[j+1,j] = 1.
        N[j+half_maxtrix,j+half_maxtrix+1] = 1.
        N[j+half_maxtrix+1,j+half_maxtrix] = 1.
        N[j,j] = -(rk**2*dy**2+2.+dy**2)
        N[j+half_maxtrix,j+half_maxtrix] = -(rk**2*dy**2+2.+dy**2)
        N[j,j+half_maxtrix]=dy**2
        N[j+half_maxtrix,j]=dy**2
        
    jo = n-3    
    M[jo,jo] = -u1[jo+1]*(rk**2*dy**2+2.+dy**2)
    M[jo,jo] += (beta*dy**2-(u1[jo+2]+u1[jo]-2.*u1[jo+1]))
    M[jo,jo] += (u1[jo+1]-u2[jo+1])*dy**2
    M[jo+half_maxtrix,jo+half_maxtrix] = -u2[jo+1]*(rk**2*dy**2+2.+dy**2)
    M[jo+half_maxtrix,jo+half_maxtrix] += (beta*dy**2-(u2[jo+2]+u2[jo]-2.*u2[jo+1]))
    M[jo+half_maxtrix,jo+half_maxtrix] += -(u1[jo+1]-u2[jo+1])*dy**2
    M[jo,jo+half_maxtrix]=u1[jo+1]*dy**2
    M[jo+half_maxtrix,jo]=u2[jo+1]*dy**2
    
    N[jo,jo] = -(rk**2*dy**2+2.+dy**2)
    N[jo+half_maxtrix,jo+half_maxtrix] = -(rk**2*dy**2+2.+dy**2)
    N[jo,jo+half_maxtrix]=dy**2
    N[jo+half_maxtrix,jo]=dy**2

    return M, N

