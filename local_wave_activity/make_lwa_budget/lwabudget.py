"""
LWA budget helper functions.

Functions include:
- lwatend: time tendency via leapfrog
- urefadv: advection by Uref
- ueadv: advection by eddies
- eddyflux_x, eddyflux_y, eddyflux_z: EP flux convergence terms
- eddyflux: EP flux
- LH: latent heating effect term
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# 1. LWA tendency
def lwatend(lwa: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
	"""Compute time tendency of LWA using leapfrog (forward/backward at ends).

	:param lwa: Local wave activity array with dimensions (time, latitude, longitude)
	:param dt: Time step
	:return: Time derivative of LWA with same shape as input
	"""
	# time-tendency of LWA
	# calculated using leapfrog except at t=0 and -1.
	lwatend: NDArray[np.floating] = np.zeros(np.shape(lwa))
	lwatend[1:-1,:,:] = (lwa[2:,:,:]-lwa[:-2,:,:])/(2*dt)
	lwatend[0,:,:] = (lwa[1,:,:]-lwa[0,:,:])/(dt)
	lwatend[-1,:,:] = (lwa[-1,:,:]-lwa[-2,:,:])/(dt)
	return lwatend


# 2. Advection by Uref
def urefadv(lwa: NDArray[np.floating], uref: NDArray[np.floating], dx: float, filt: bool=True) -> NDArray[np.floating]:
	"""Compute advection of LWA by reference wind Uref.

	:param lwa: LWA field (time, latitude, longitude)
	:param uref: Reference zonal wind (time, latitude)
	:param dx: Zonal grid spacing
	:param filt: If True, apply 1-2-1 temporal filter to the result
	:return: Tendency due to Uref advection
	"""
	# advection of LWA by uref
	# second-order finite differencing on periodic boundary
	# filt: if True, apply 1-2-1 filter in time
	lwagradx: NDArray[np.floating] = (np.roll(lwa,-1,axis=2)-np.roll(lwa,1,axis=2))/(2*dx)
	out: NDArray[np.floating] = -lwagradx*uref[:,:,np.newaxis]
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return out


# 3. Advection by ue
def ueadv(
	q: NDArray[np.floating],
	qref: NDArray[np.floating],
	u: NDArray[np.floating],
	uref: NDArray[np.floating],
	dx: float,
	dy: float,
	filt: bool=True,
) -> NDArray[np.floating]:
	"""Compute advection of PV anomaly by eddy wind contribution.

	:param q: Full PV (time, latitude, longitude)
	:param qref: Reference PV (time, latitude)
	:param u: Full zonal wind (time, latitude, longitude)
	:param uref: Reference zonal wind (time, latitude)
	:param dx: Zonal grid spacing
	:param dy: Meridional grid spacing
	:param filt: If True, apply 1-2-1 temporal filter
	:return: Tendency term from eddy advection
	"""
	# advection of qe by ue d dx int^{eta}_{0} ue qe dy'
	# second-order finite differencing on periodic boundary
	# filt: if True, apply 1-2-1 filter in time
	tn,yn,xn = np.shape(q)
	Iuq: NDArray[np.floating] = np.zeros(np.shape(q))
	for t in range(tn):
		for y1 in range(yn):
			q_e = q[t,:,:]-qref[t,y1]
			u_e = u[t,:,:]-uref[t,y1]
			for x in range(xn):
				for y2 in range(yn):
					if y2 <  y1 and q_e[y2,x] > 0:
						Iuq[t,y1,x]+=u_e[y2,x]*q_e[y2,x]*dy
					if y2 >= y1 and q_e[y2,x] <= 0:
						Iuq[t,y1,x]+=u_e[y2,x]*q_e[y2,x]*(-dy)
	out: NDArray[np.floating] = (np.roll(Iuq,-1,axis=2)-np.roll(Iuq,1,axis=2))/(2*dx)
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return -out
	

# 4 zonal eddy flux convergence
def eddyflux_x(ue: NDArray[np.floating], ve: NDArray[np.floating], dx: float, filt: bool=True) -> NDArray[np.floating]:
	"""Compute zonal EP flux convergence, -1/2 d/dx (v^2 - u^2).

	:param ue: Eddy zonal wind (time, latitude, longitude)
	:param ve: Eddy meridional wind (time, latitude, longitude)
	:param dx: Zonal grid spacing
	:param filt: If True, apply 1-2-1 temporal filter
	:return: Zonal eddy flux convergence term
	"""
	# zonal eddy flux convergence, -1/2 d/dx (v^2 - u^2)
	# second-order finite differencing on periodic boundary
	# filt: if True, apply 1-2-1 filter in time
	v2_u2: NDArray[np.floating] = 0.5*(ve[:,:,:]**2-ue[:,:,:]**2)
	out: NDArray[np.floating] = -(np.roll(v2_u2,-1,axis=2)-np.roll(v2_u2,1,axis=2))/(2*dx)
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return out

# 5 meridional eddy flux convergence
def eddyflux_y(ue: NDArray[np.floating], ve: NDArray[np.floating], dy: float, filt: bool=True) -> NDArray[np.floating]:
	"""Compute meridional EP flux convergence, d/dy (uv), with zero at boundaries.

	:param ue: Eddy zonal wind
	:param ve: Eddy meridional wind
	:param dy: Meridional grid spacing
	:param filt: If True, apply 1-2-1 temporal filter
	:return: Meridional eddy flux convergence term
	"""
	# meridional eddy flux convergence,  d/dy (uv)
	# second-order finite differencing, uv = 0 added at north and south
	# filt: if True, apply 1-2-1 filter in time
	uv: NDArray[np.floating] = np.pad(ue[:,:,:]*ve[:,:,:],((0,0),(1,1),(0,0)),mode='constant',constant_values=0)
	out: NDArray[np.floating] = (uv[:,2:,:]-uv[:,:-2,:])/(2*dy)
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return out


# 6 vertical eddy flux convergence (heat flux)
def eddyflux_z(ve: NDArray[np.floating], te: NDArray[np.floating], Ld: float, filt: bool=True) -> NDArray[np.floating]:
	"""Compute vertical eddy heat flux convergence, vT/Ld**2.

	:param ve: Eddy meridional wind
	:param te: Eddy temperature
	:param Ld: Deformation radius
	:param filt: If True, apply 1-2-1 temporal filter
	:return: Vertical eddy flux convergence term
	"""
	# heat flux convergence,  vT/Ld**2
	# filt: if True, apply 1-2-1 filter in time
	out: NDArray[np.floating] = ve*te/(Ld**2)
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return out


# 7 all EP flux
def eddyflux(ve: NDArray[np.floating], qe: NDArray[np.floating], filt: bool=True) -> NDArray[np.floating]:
	"""Compute EP flux term -ve*qe.

	:param ve: Eddy meridional wind
	:param qe: Eddy PV (or tracer) anomaly
	:param filt: If True, apply 1-2-1 temporal filter
	:return: EP flux term
	"""
	# EP flux
	out: NDArray[np.floating] = -ve*qe
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return out

# 8. Effect of LH
def LH(
	p: NDArray[np.floating],
	q: NDArray[np.floating],
	qref: NDArray[np.floating],
	L: float,
	dx: float,
	dy: float,
	filt: bool=True,
) -> NDArray[np.floating]:
	"""Compute latent heating contribution integrated meridionally with sign selection.

	:param p: Precipitation field (time, latitude, longitude)
	:param q: Full PV (time, latitude, longitude)
	:param qref: Reference PV (time, latitude)
	:param L: Latent heating parameter
	:param dx: Zonal grid spacing (unused but kept for consistency)
	:param dy: Meridional grid spacing
	:param filt: If True, apply 1-2-1 temporal filter
	:return: Latent heating term
	"""
	# effect of LH
	# second-order finite differencing on periodic boundary
	# filt: if True, apply 1-2-1 filter in time
	tn,yn,xn = np.shape(p)
	out: NDArray[np.floating] = np.zeros(np.shape(p))
	for t in range(tn):
		for y1 in range(yn):
			q_e = q[t,:,:]-qref[t,y1]
			for x in range(xn):
				for y2 in range(yn):
					if y2 <  y1 and q_e[y2,x] > 0:
						out[t,y1,x]+=L*p[t,y2,x]*dy
					if y2 >= y1 and q_e[y2,x] <= 0:
						out[t,y1,x]+=L*p[t,y2,x]*(-dy)
	if filt == True:	out[1:-1,:]=out[:-2,:]*0.25+out[1:-1,:]*0.5+out[2:,:]*0.25
	return -out 

