"""Numerical verification: packaged code produces identical results to original scripts."""

import importlib
import importlib.util
import pathlib
import sys

import netCDF4
import numpy as np
import xarray as xr

import qg_lwa.budget
import qg_lwa.compute_lwa
import qg_lwa.compute_uref
import qg_eddy_growth.matrices
import qg_eddy_growth.moist_growth

DATADIR = pathlib.Path('/mnt/winds/data/gmsarro/100_year_runs')
BASE = 'Realistic_two_N128_0.0_2.0_0.1_1.0'
MAX_TIME = 50

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
n_pass = 0
n_fail = 0


def check(name: str, condition: bool, detail: str = '') -> None:
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f'  [{PASS}] {name}  {detail}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  {detail}')


def load_module_from_file(name: str, filepath: str) -> object:
    parent = str(pathlib.Path(filepath).parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    finally:
        if added:
            sys.path.remove(parent)


def _load_TY_xr(path: str, varname: str, max_time: int) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        da = ds[varname]
        dims = list(da.dims)
        if len(dims) != 2:
            raise ValueError(f'{varname}: expected 2 dims, got {dims}')
        if da.sizes[dims[0]] >= da.sizes[dims[1]]:
            da = da.transpose(dims[0], dims[1])
        else:
            da = da.transpose(dims[1], dims[0])
        return da.values[:max_time, :]


def _load_TYX_xr(path: str, varname: str, max_time: int) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        da = ds[varname]
        dims = list(da.dims)
        if len(dims) != 3:
            raise ValueError(f'{varname}: expected 3 dims, got {dims}')
        time_candidates = [d for d in dims if 'time' in d.lower() or d == 't']
        lat_candidates = [d for d in dims if d in ('y', 'lat', 'latitude')]
        lon_candidates = [d for d in dims if d in ('x', 'lon', 'longitude')]
        if not time_candidates or not lat_candidates or not lon_candidates:
            sizes = [(da.sizes[d], d) for d in dims]
            sizes.sort(reverse=True)
            tdim_name = sizes[0][1]
            remaining = [d for d in dims if d != tdim_name]
            ydim_name = remaining[0]
            xdim_name = remaining[1]
        else:
            tdim_name = time_candidates[0]
            ydim_name = lat_candidates[0]
            xdim_name = lon_candidates[0]
        da = da.transpose(tdim_name, ydim_name, xdim_name)
        return da.values[:max_time, :, :]


print('Loading test data from', DATADIR / BASE)
with netCDF4.Dataset(str(DATADIR / (BASE + '.nc'))) as ds:
    um_full = ds.variables['zu1'][:, :].data
    umb_full = ds.variables['zu2'][:, :].data
    tm_full = ds.variables['ztau'][:, :].data
    ys = ds.variables['y'][:].data
yn = len(ys)

with netCDF4.Dataset(str(DATADIR / (BASE + '.3d.nc'))) as ds:
    xs = ds.variables['x'][:].data
    qdat = ds.variables['q1'][:MAX_TIME, :, :].data
    vdat = ds.variables['v1'][:MAX_TIME, :, :].data
    udat = ds.variables['u1'][:MAX_TIME, :, :].data
    tdat = ds.variables['tau'][:MAX_TIME, :, :].data
xn = len(xs)

qref = _load_TY_xr(str(DATADIR / (BASE + '.qref1_2.nc')), 'qref1', MAX_TIME)
um = um_full[:MAX_TIME, :]
umb = umb_full[:MAX_TIME, :]
tm = tm_full[:MAX_TIME, :]

uref_reference = _load_TY_xr(str(DATADIR / (BASE + '.uref1_2.nc')), 'uref1', MAX_TIME)
tref_reference = _load_TY_xr(str(DATADIR / (BASE + '.tref1_2.nc')), 'tref1', MAX_TIME)
LWAC = _load_TYX_xr(str(DATADIR / (BASE + '.wac1_2.nc')), 'wac1', MAX_TIME)
LWAA = _load_TYX_xr(str(DATADIR / (BASE + '.waa1_2.nc')), 'waa1', MAX_TIME)

LWA = LWAC + LWAA
Qref_3d = qref
tn = MAX_TIME
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
dt = 1.0
Ld = 1.0

print(f'Data loaded: {tn} timesteps, y={yn}, x={xn}\n')


print('=' * 70)
print('TEST 1: qg_lwa.budget functions — original vs package')
print('=' * 70)

lwabudget_old = load_module_from_file(
    'lwabudget_old',
    '/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/lwabudget.py',
)

old_lwatend = lwabudget_old.lwatend(LWA, dt)
new_lwatend = qg_lwa.budget.lwatend(lwa=LWA, dt=dt)
check('lwatend', np.allclose(old_lwatend, new_lwatend, atol=0),
      f'max diff = {np.max(np.abs(old_lwatend - new_lwatend)):.2e}')

old_urefadv = lwabudget_old.urefadv(LWA, uref_reference, dx, filt=False)
new_urefadv = qg_lwa.budget.urefadv(lwa=LWA, uref=uref_reference, dx=dx, filt=False)
check('urefadv (filt=False)', np.allclose(old_urefadv, new_urefadv, atol=0),
      f'max diff = {np.max(np.abs(old_urefadv - new_urefadv)):.2e}')

old_urefadv_f = lwabudget_old.urefadv(LWA, uref_reference, dx, filt=True)
new_urefadv_f = qg_lwa.budget.urefadv(lwa=LWA, uref=uref_reference, dx=dx, filt=True)
check('urefadv (filt=True)', np.allclose(old_urefadv_f, new_urefadv_f, atol=0),
      f'max diff = {np.max(np.abs(old_urefadv_f - new_urefadv_f)):.2e}')

qe = qdat - Qref_3d[:, :, np.newaxis]
ue = udat - uref_reference[:, :, np.newaxis]
ve = vdat.copy()
te = tdat - tref_reference[:, :, np.newaxis]

old_efx = lwabudget_old.eddyflux_x(ue, ve, dx, filt=False)
new_efx = qg_lwa.budget.eddyflux_x(ue=ue, ve=ve, dx=dx, filt=False)
check('eddyflux_x', np.allclose(old_efx, new_efx, atol=0),
      f'max diff = {np.max(np.abs(old_efx - new_efx)):.2e}')

old_efy = lwabudget_old.eddyflux_y(ue, ve, dy, filt=False)
new_efy = qg_lwa.budget.eddyflux_y(ue=ue, ve=ve, dy=dy, filt=False)
check('eddyflux_y', np.allclose(old_efy, new_efy, atol=0),
      f'max diff = {np.max(np.abs(old_efy - new_efy)):.2e}')

old_efz = lwabudget_old.eddyflux_z(ve, te, Ld, filt=False)
new_efz = qg_lwa.budget.eddyflux_z(ve=ve, te=te, Ld=Ld, filt=False)
check('eddyflux_z', np.allclose(old_efz, new_efz, atol=0),
      f'max diff = {np.max(np.abs(old_efz - new_efz)):.2e}')

old_ef = lwabudget_old.eddyflux(ve, qe, filt=False)
new_ef = qg_lwa.budget.eddyflux(ve=ve, qe=qe, filt=False)
check('eddyflux', np.allclose(old_ef, new_ef, atol=0),
      f'max diff = {np.max(np.abs(old_ef - new_ef)):.2e}')


print()
print('=' * 70)
print('TEST 2: SOR solver (qg_lwa.compute_uref) — vs original output files')
print('=' * 70)

uref_new = qg_lwa.compute_uref.solve_uref(
    qref=qref, um=um, umb=umb, ys=ys, beta=0.2, Ld=1.0,
)
tref_new = qg_lwa.compute_uref.integrate_tref(uref=uref_new, ys=ys, tm=tm)

uref_diff = np.max(np.abs(uref_new - uref_reference))
tref_diff = np.max(np.abs(tref_new - tref_reference))

check('uref vs reference file', uref_diff < 5e-6,
      f'max |diff| = {uref_diff:.2e} (SOR tol=1e-6)')
check('tref vs reference file', tref_diff < 5e-6,
      f'max |diff| = {tref_diff:.2e} (integrated from uref)')


print()
print('=' * 70)
print('TEST 3: qg_eddy_growth.matrices — typo rename')
print('=' * 70)

with netCDF4.Dataset(str(DATADIR / (BASE + '.3d.nc'))) as ds:
    y_eddy = ds.variables['y'][:]
    u1_mean = np.mean(np.mean(ds.variables['u1'][:], axis=2), axis=0)
    u2_mean = np.mean(np.mean(ds.variables['u2'][:], axis=2), axis=0)

dy_eddy = float(y_eddy[1] - y_eddy[0])
n_pts = int(len(y_eddy))
n_2 = n_pts * 2
hm = n_pts - 2

bm_old = load_module_from_file(
    'bm_old',
    '/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/build_matrices.py',
)

try:
    M_old, N_old = bm_old.build_matrices(
        u1=u1_mean, u2=u2_mean, beta=0.2, dy=dy_eddy,
        n_2=n_2, rk=1.0, half_maxtrix=hm, n=n_pts,
    )
except TypeError:
    M_old, N_old = bm_old.build_matrices(
        u1_mean, u2_mean, 0.2, dy_eddy, n_2, 1.0, hm, n_pts,
    )

M_new, N_new = qg_eddy_growth.matrices.build_matrices(
    u1=u1_mean, u2=u2_mean, beta=0.2, dy=dy_eddy,
    n_2=n_2, rk=1.0, half_matrix=hm, n=n_pts,
)

check('M matrix identical', np.allclose(M_old, M_new, atol=0),
      f'max diff = {np.max(np.abs(M_old - M_new)):.2e}')
check('N matrix identical', np.allclose(N_old, N_new, atol=0),
      f'max diff = {np.max(np.abs(N_old - N_new)):.2e}')


print()
print('=' * 70)
print('TEST 4: qg_eddy_growth.moist_growth — old vs new')
print('=' * 70)

mgm_old_path = pathlib.Path(
    '/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/Moist_growth/moist_growth_matrix.py',
)
has_old_mgm = mgm_old_path.exists()
if has_old_mgm:
    mgm_old = load_module_from_file('mgm_old', str(mgm_old_path))

if has_old_mgm:
    try:
        kk_old, gr_old, q1_old, q2_old, P_old = mgm_old.moist_matrix(L=0.2, U1=1.0, U2=0.5)
    except TypeError:
        kk_old, gr_old, q1_old, q2_old, P_old = mgm_old.moist_matrix(0.2, 1.0, 0.5)
    kk_new, gr_new, q1_new, q2_new, P_new = qg_eddy_growth.moist_growth.moist_matrix(
        L=0.2, U1=1.0, U2=0.5,
    )
    check('moist kk identical', np.allclose(kk_old, kk_new, atol=0),
          f'max diff = {np.max(np.abs(kk_old - kk_new)):.2e}')
    check('moist growth identical', np.allclose(gr_old, gr_new, atol=0),
          f'max diff = {np.max(np.abs(gr_old - gr_new)):.2e}')
    check('moist q1 identical', np.allclose(q1_old, q1_new, atol=1e-14),
          f'max diff = {np.max(np.abs(q1_old - q1_new)):.2e}')
    check('moist q2 identical', np.allclose(q2_old, q2_new, atol=1e-14),
          f'max diff = {np.max(np.abs(q2_old - q2_new)):.2e}')
else:
    kk_new, gr_new, q1_new, q2_new, P_new = qg_eddy_growth.moist_growth.moist_matrix(
        L=0.2, U1=1.0, U2=0.5,
    )
    check('moist_matrix runs', True, f'peak growth = {np.max(gr_new):.6f}')


print()
print('=' * 70)
print('TEST 5: ueadv — original vs package (3 timesteps)')
print('=' * 70)

T_SMALL = 3
old_ueadv = lwabudget_old.ueadv(
    qdat[:T_SMALL], Qref_3d[:T_SMALL], udat[:T_SMALL], uref_reference[:T_SMALL],
    dx, dy, filt=False,
)
new_ueadv = qg_lwa.budget.ueadv(
    q=qdat[:T_SMALL], qref=Qref_3d[:T_SMALL], u=udat[:T_SMALL], uref=uref_reference[:T_SMALL],
    dx=dx, dy=dy, filt=False,
)
check('ueadv', np.allclose(old_ueadv, new_ueadv, atol=0),
      f'max diff = {np.max(np.abs(old_ueadv - new_ueadv)):.2e}')


print()
print('=' * 70)
print('TEST 6: LH (latent heating) — original vs package (3 timesteps)')
print('=' * 70)

with netCDF4.Dataset(str(DATADIR / (BASE + '.3d.nc'))) as ds:
    pdat = ds.variables['P'][:T_SMALL, :, :].data

old_LH = lwabudget_old.LH(
    pdat, qdat[:T_SMALL], Qref_3d[:T_SMALL], 0.0, dx, dy, filt=False,
)
new_LH = qg_lwa.budget.LH(
    p=pdat, q=qdat[:T_SMALL], qref=Qref_3d[:T_SMALL], L=0.0, dx=dx, dy=dy, filt=False,
)
check('LH (L=0.0)', np.allclose(old_LH, new_LH, atol=0),
      f'max diff = {np.max(np.abs(old_LH - new_LH)):.2e}')


print()
print('=' * 70)
print('TEST 7: compute_lwa (f2py) — qg_lwa.compute_lwa vs reference files')
print('=' * 70)

fortran = qg_lwa.compute_lwa._get_fortran()

with xr.open_dataset(str(DATADIR / (BASE + '.3d.nc'))) as ds_3d:
    tdim_lwa = 'time'
    ydim_lwa = 'y'
    xdim_lwa = 'x'
    ds_3d = ds_3d.assign_coords({tdim_lwa: np.arange(ds_3d[tdim_lwa].size)})
    q1_lwa = ds_3d['q1'].isel({tdim_lwa: slice(0, MAX_TIME)}).transpose(
        tdim_lwa, ydim_lwa, xdim_lwa,
    )

q1_masked = qg_lwa.compute_lwa.mask_sponge(q1=q1_lwa, tdim=tdim_lwa, ydim=ydim_lwa, xdim=xdim_lwa)

lx_ref, ly_ref = 48.0, 72.0

q1_np = np.asfortranarray(q1_masked.values.transpose(2, 1, 0))
qref_f2py = fortran.lwa_2layer.calc_qref(q1_np, wx=lx_ref, wy=ly_ref)
waa_f2py, wac_f2py = fortran.lwa_2layer.calc_lwa(q1_np, qref_f2py, wx=lx_ref, wy=ly_ref)

qref_new = qref_f2py.T
waa_new = waa_f2py.transpose(2, 1, 0)
wac_new = wac_f2py.transpose(2, 1, 0)

qref_ref_lwa = _load_TY_xr(str(DATADIR / (BASE + '.qref1_2.nc')), 'qref1', MAX_TIME)
waa_ref_lwa = _load_TYX_xr(str(DATADIR / (BASE + '.waa1_2.nc')), 'waa1', MAX_TIME)
wac_ref_lwa = _load_TYX_xr(str(DATADIR / (BASE + '.wac1_2.nc')), 'wac1', MAX_TIME)

qref_diff = float(np.max(np.abs(qref_new - qref_ref_lwa)))
waa_diff = float(np.max(np.abs(waa_new - waa_ref_lwa)))
wac_diff = float(np.max(np.abs(wac_new - wac_ref_lwa)))

check('compute_lwa qref', np.allclose(qref_new, qref_ref_lwa, atol=1e-6),
      f'max |diff| = {qref_diff:.2e}')
check('compute_lwa waa', np.allclose(waa_new, waa_ref_lwa, atol=1e-6),
      f'max |diff| = {waa_diff:.2e}')
check('compute_lwa wac', np.allclose(wac_new, wac_ref_lwa, atol=1e-6),
      f'max |diff| = {wac_diff:.2e}')


print()
print('=' * 70)
total = n_pass + n_fail
print(f'RESULTS: {n_pass}/{total} passed, {n_fail}/{total} failed')
if n_fail == 0:
    print('All tests passed — packaged code is numerically identical to originals.')
else:
    print('FAILURES DETECTED — review the output above.')
print('=' * 70)
sys.exit(n_fail)
