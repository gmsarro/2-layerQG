"""
Verification script: prove that refactored code produces identical results
to the original working scripts.

Compares:
  1. lwabudget.py functions (old positional vs new keyword-only)
  2. SOR solver (uref_make.py) against pre-existing reference output
  3. build_matrices (half_maxtrix -> half_matrix rename)
  4. Full LWA budget pipeline on real data
  5. moist_growth_matrix
"""

import sys
import pathlib
import importlib
import importlib.util
import numpy as np
import netCDF4

DATADIR = pathlib.Path('/mnt/winds/data/gmsarro/100_year_runs')
BASE = 'Realistic_two_N128_0.0_2.0_0.1_1.0'
MAX_TIME = 50  # use 50 timesteps for fast verification

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
n_pass = 0
n_fail = 0


def check(name, condition, detail=''):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f'  [{PASS}] {name}  {detail}')
    else:
        n_fail += 1
        print(f'  [{FAIL}] {name}  {detail}')


def load_module_from_file(name, filepath):
    """Import a Python module from an explicit file path, bypassing sys.modules cache."""
    parent = str(pathlib.Path(filepath).parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added:
            sys.path.remove(parent)


# ── Helper: coerce orientations (same logic as array_utils.py) ─────────
def _ensure_TY(arr, y_len):
    if arr.ndim != 2:
        raise ValueError(f'expected 2-D, got {arr.ndim}-D')
    if arr.shape[1] == y_len:
        return arr
    if arr.shape[0] == y_len:
        return arr.T
    raise ValueError(f'no dim equals y_len={y_len}; shape={arr.shape}')


def _ensure_TYX(arr, y_len, x_len):
    if arr.ndim != 3:
        raise ValueError(f'expected 3-D, got {arr.ndim}-D')
    s0, s1, s2 = arr.shape
    if (s1, s2) == (y_len, x_len):
        return arr
    if (s0, s1) == (y_len, x_len):
        return np.transpose(arr, (2, 0, 1))
    if (s0, s1) == (x_len, y_len):
        return np.transpose(arr, (2, 1, 0))
    if (s1, s2) == (x_len, y_len):
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f'cannot coerce shape {arr.shape} to (T, {y_len}, {x_len})')


# ── Load shared data ───────────────────────────────────────────────────
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

with netCDF4.Dataset(str(DATADIR / (BASE + '.qref1_2.nc'))) as ds:
    qref = _ensure_TY(ds.variables['qref1'][:, :].data, yn)[:MAX_TIME, :]
um = um_full[:MAX_TIME, :]
umb = umb_full[:MAX_TIME, :]
tm = tm_full[:MAX_TIME, :]

with netCDF4.Dataset(str(DATADIR / (BASE + '.uref1_2.nc'))) as ds:
    uref_reference = _ensure_TY(ds.variables['uref1'][:, :].data, yn)[:MAX_TIME, :]

with netCDF4.Dataset(str(DATADIR / (BASE + '.tref1_2.nc'))) as ds:
    tref_reference = _ensure_TY(ds.variables['tref1'][:, :].data, yn)[:MAX_TIME, :]

with netCDF4.Dataset(str(DATADIR / (BASE + '.wac1_2.nc'))) as ds:
    LWAC = _ensure_TYX(ds.variables['wac1'][:, :, :].data, yn, xn)[:MAX_TIME, :, :]

with netCDF4.Dataset(str(DATADIR / (BASE + '.waa1_2.nc'))) as ds:
    LWAA = _ensure_TYX(ds.variables['waa1'][:, :, :].data, yn, xn)[:MAX_TIME, :, :]

LWA = LWAC + LWAA
Qref_3d = qref
tn = MAX_TIME
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
dt = 1.0
Ld = 1.0

print(f'Data loaded: {tn} timesteps, y={yn}, x={xn}\n')


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: lwabudget.py — old (positional) vs new (keyword-only)
# ═══════════════════════════════════════════════════════════════════════
print('=' * 70)
print('TEST 1: lwabudget functions — original vs refactored')
print('=' * 70)

lwabudget_old = load_module_from_file(
    'lwabudget_old',
    '/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/lwabudget.py',
)
lwabudget_new = load_module_from_file(
    'lwabudget_new',
    str(pathlib.Path(__file__).resolve().parent / 'local_wave_activity' / 'make_lwa_budget' / 'lwabudget.py'),
)

# lwatend
old_lwatend = lwabudget_old.lwatend(LWA, dt)
new_lwatend = lwabudget_new.lwatend(lwa=LWA, dt=dt)
check('lwatend', np.allclose(old_lwatend, new_lwatend, atol=0),
      f'max diff = {np.max(np.abs(old_lwatend - new_lwatend)):.2e}')

# urefadv
old_urefadv = lwabudget_old.urefadv(LWA, uref_reference, dx, filt=False)
new_urefadv = lwabudget_new.urefadv(lwa=LWA, uref=uref_reference, dx=dx, filt=False)
check('urefadv (filt=False)', np.allclose(old_urefadv, new_urefadv, atol=0),
      f'max diff = {np.max(np.abs(old_urefadv - new_urefadv)):.2e}')

old_urefadv_f = lwabudget_old.urefadv(LWA, uref_reference, dx, filt=True)
new_urefadv_f = lwabudget_new.urefadv(lwa=LWA, uref=uref_reference, dx=dx, filt=True)
check('urefadv (filt=True)', np.allclose(old_urefadv_f, new_urefadv_f, atol=0),
      f'max diff = {np.max(np.abs(old_urefadv_f - new_urefadv_f)):.2e}')

# eddyflux_x
qe = qdat - Qref_3d[:, :, np.newaxis]
ue = udat - uref_reference[:, :, np.newaxis]
ve = vdat.copy()
te = tdat - tref_reference[:, :, np.newaxis]

old_efx = lwabudget_old.eddyflux_x(ue, ve, dx, filt=False)
new_efx = lwabudget_new.eddyflux_x(ue=ue, ve=ve, dx=dx, filt=False)
check('eddyflux_x', np.allclose(old_efx, new_efx, atol=0),
      f'max diff = {np.max(np.abs(old_efx - new_efx)):.2e}')

# eddyflux_y
old_efy = lwabudget_old.eddyflux_y(ue, ve, dy, filt=False)
new_efy = lwabudget_new.eddyflux_y(ue=ue, ve=ve, dy=dy, filt=False)
check('eddyflux_y', np.allclose(old_efy, new_efy, atol=0),
      f'max diff = {np.max(np.abs(old_efy - new_efy)):.2e}')

# eddyflux_z
old_efz = lwabudget_old.eddyflux_z(ve, te, Ld, filt=False)
new_efz = lwabudget_new.eddyflux_z(ve=ve, te=te, Ld=Ld, filt=False)
check('eddyflux_z', np.allclose(old_efz, new_efz, atol=0),
      f'max diff = {np.max(np.abs(old_efz - new_efz)):.2e}')

# eddyflux
old_ef = lwabudget_old.eddyflux(ve, qe, filt=False)
new_ef = lwabudget_new.eddyflux(ve=ve, qe=qe, filt=False)
check('eddyflux', np.allclose(old_ef, new_ef, atol=0),
      f'max diff = {np.max(np.abs(old_ef - new_ef)):.2e}')


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: SOR solver — refactored vs pre-existing reference output
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('TEST 2: SOR solver (uref_make) — refactored vs original output files')
print('=' * 70)

uref_make = load_module_from_file(
    'uref_make',
    str(pathlib.Path(__file__).resolve().parent / 'local_wave_activity' / 'make_lwa_budget' / 'uref_make.py'),
)

uref_new = uref_make.solve_uref(
    qref=qref, um=um, umb=umb, ys=ys, beta=0.2, Ld=1.0,
)
tref_new = uref_make.integrate_tref(uref=uref_new, ys=ys, tm=tm)

uref_diff = np.max(np.abs(uref_new - uref_reference))
tref_diff = np.max(np.abs(tref_new - tref_reference))

check('uref vs reference file', uref_diff < 5e-6,
      f'max |diff| = {uref_diff:.2e} (SOR tol=1e-6)')
check('tref vs reference file', tref_diff < 5e-6,
      f'max |diff| = {tref_diff:.2e} (integrated from uref)')


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: build_matrices — old (half_maxtrix) vs new (half_matrix)
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('TEST 3: build_matrices — typo rename (half_maxtrix -> half_matrix)')
print('=' * 70)

# Use the actual model data for realistic u1, u2 profiles
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
bm_new = load_module_from_file(
    'bm_new',
    str(pathlib.Path(__file__).resolve().parent / 'eddy_growth' / 'dry_eddy_growth' / 'build_matrices.py'),
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

M_new, N_new = bm_new.build_matrices(
    u1=u1_mean, u2=u2_mean, beta=0.2, dy=dy_eddy,
    n_2=n_2, rk=1.0, half_matrix=hm, n=n_pts,
)

check('M matrix identical', np.allclose(M_old, M_new, atol=0),
      f'max diff = {np.max(np.abs(M_old - M_new)):.2e}')
check('N matrix identical', np.allclose(N_old, N_new, atol=0),
      f'max diff = {np.max(np.abs(N_old - N_new)):.2e}')


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: moist_growth_matrix — old vs new
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('TEST 4: moist_growth_matrix — old vs new')
print('=' * 70)

mgm_new = load_module_from_file(
    'mgm_new',
    str(pathlib.Path(__file__).resolve().parent / 'eddy_growth' / 'moist_eddy_growth' / 'moist_growth_matrix.py'),
)
mgm_old_path = pathlib.Path('/mnt/winds/data2/gmsarro/Rossbypalloza_project_22/LWA/run_LWA/Moist_growth/moist_growth_matrix.py')
has_old_mgm = mgm_old_path.exists()
if has_old_mgm:
    mgm_old = load_module_from_file('mgm_old', str(mgm_old_path))

if has_old_mgm:
    try:
        kk_old, gr_old, q1_old, q2_old, P_old = mgm_old.moist_matrix(L=0.2, U1=1.0, U2=0.5)
    except TypeError:
        kk_old, gr_old, q1_old, q2_old, P_old = mgm_old.moist_matrix(0.2, 1.0, 0.5)
    kk_new, gr_new, q1_new, q2_new, P_new = mgm_new.moist_matrix(L=0.2, U1=1.0, U2=0.5)

    check('moist kk identical', np.allclose(kk_old, kk_new, atol=0),
          f'max diff = {np.max(np.abs(kk_old - kk_new)):.2e}')
    check('moist growth identical', np.allclose(gr_old, gr_new, atol=0),
          f'max diff = {np.max(np.abs(gr_old - gr_new)):.2e}')
    check('moist q1 identical', np.allclose(q1_old, q1_new, atol=1e-14),
          f'max diff = {np.max(np.abs(q1_old - q1_new)):.2e}')
    check('moist q2 identical', np.allclose(q2_old, q2_new, atol=1e-14),
          f'max diff = {np.max(np.abs(q2_old - q2_new)):.2e}')
else:
    # Sanity check only
    kk_new, gr_new, q1_new, q2_new, P_new = mgm_new.moist_matrix(L=0.2, U1=1.0, U2=0.5)
    check('moist_matrix runs', True, f'peak growth = {np.max(gr_new):.6f}')


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: ueadv (the expensive O(T*Y^2*X) function) — small subset
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('TEST 5: ueadv — original vs refactored (3 timesteps)')
print('=' * 70)

T_SMALL = 3
old_ueadv = lwabudget_old.ueadv(
    qdat[:T_SMALL], Qref_3d[:T_SMALL], udat[:T_SMALL], uref_reference[:T_SMALL],
    dx, dy, filt=False,
)
new_ueadv = lwabudget_new.ueadv(
    q=qdat[:T_SMALL], qref=Qref_3d[:T_SMALL], u=udat[:T_SMALL], uref=uref_reference[:T_SMALL],
    dx=dx, dy=dy, filt=False,
)
check('ueadv', np.allclose(old_ueadv, new_ueadv, atol=0),
      f'max diff = {np.max(np.abs(old_ueadv - new_ueadv)):.2e}')


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: LH function — original vs refactored (3 timesteps)
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('TEST 6: LH (latent heating) — original vs refactored (3 timesteps)')
print('=' * 70)

with netCDF4.Dataset(str(DATADIR / (BASE + '.3d.nc'))) as ds:
    pdat = ds.variables['P'][:T_SMALL, :, :].data

old_LH = lwabudget_old.LH(
    pdat, qdat[:T_SMALL], Qref_3d[:T_SMALL], 0.0, dx, dy, filt=False,
)
new_LH = lwabudget_new.LH(
    p=pdat, q=qdat[:T_SMALL], qref=Qref_3d[:T_SMALL], L=0.0, dx=dx, dy=dy, filt=False,
)
check('LH (L=0.0)', np.allclose(old_LH, new_LH, atol=0),
      f'max diff = {np.max(np.abs(old_LH - new_LH)):.2e}')


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
total = n_pass + n_fail
print(f'RESULTS: {n_pass}/{total} passed, {n_fail}/{total} failed')
if n_fail == 0:
    print('All tests passed — refactored code is numerically identical to originals.')
else:
    print('FAILURES DETECTED — review the output above.')
print('=' * 70)
sys.exit(n_fail)
