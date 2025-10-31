import numpy as np
# np.seterr(all='raise')
np.seterr(under='ignore', invalid='raise', divide='raise', over='raise')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import kv  # K_v BESSEL FUNCTIONS
from scipy.special import zeta, polygamma, factorial
import matplotlib as mpl
from mpmath import zeta
from itertools import cycle
from scipy.special import kve


###
GSTAR = 90.0
GSTAR_S = 90.0
ME = 0.000511e-3  # GeV
ALPHA_EM = 1.0/137.0
MPL   = 1.2209e19


LOG_TINY = -745.0      # ~ log(np.finfo(float).tiny)
VAL_FLOOR = 1e-300     # positive floor
######

def gstar_interp(T, path = "/Users/charlottemyers/projects/ctp/heffBW.dat"):
    #if T < T_data.min() or T > T_data.max():
        #return GSTAR
    data = np.loadtxt(path)
    T_data = data[:,0]
    g_eff = data[:,1]

    return np.interp(T, T_data, g_eff)

def m_sv_interp(T, path = "/Users/charlottemyers/projects/ctp/thermal_xsec.txt"):
    m = []
    sv = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                m_str, sv_str = line.strip().split(",")
                m.append(float(m_str))
                sv.append(float(sv_str))
    return np.interp(T, m, sv)


def returnk (T, path = "/Users/charlottemyers/projects/ctp/heffBW.dat"):
    return T



# ----------------------------
# Entropy functions
# ----------------------------
def gstar_SM(T, t_dep = False):
    if t_dep:
        gstar_s = gstar_interp(T)
    else:
        gstar_s = GSTAR_S
    return  gstar_s
    #return GSTAR

def gstars_SM(T, t_dep = False):
    return gstar_SM(T, t_dep)
    #return GSTAR_S


def dln_gstars_SM_dlnT(T, t_dep = False, delta=1e-6):
    """Numerical derivative of ln(g*_s) w.r.t. ln(T)."""
    if t_dep:
        gstar_plus = gstar_interp(T * (1.0 + delta))
        gstar_minus = gstar_interp(T * (1.0 - delta))
        return (np.log(gstar_plus) - np.log(gstar_minus)) / (2.0 * np.log(1.0 + delta))
    else:
        return 0


# ----------------------------
# Helper thermo stuff
# ----------------------------


def dx_dt(T, H, m, t_dep = False):
    """dx/dt for x = mchi / T."""
    x = m / T
    corr = 1.0 + (1.0/3.0) * dln_gstars_SM_dlnT(T, t_dep)
    return H * x / corr


def neq(m, g, T):
    """equilibrium number density at temperature T - Use bessels"""
    x = m / T
    return g * (m**2*T) / (2.0*np.pi**2) * kv(2, x)


def rho_i_exact(n_i, m_i, Th):
    # source: 2504.00077, eqn 2.15
    Th   = np.maximum(Th, VAL_FLOOR)        # avoid z = inf / division by zero
    z    = m_i / Th
    # Use scaled Kv: Kv(z) = e^{-z} kve(order, z);  ratio is stable as kve(1)/kve(2)
    k2   = kve(2, z)
    ratio = kve(1, z) / np.maximum(k2, VAL_FLOOR)
    return n_i * (m_i * ratio + 3.0 * Th)

def P_i_exact(n_i, Th):
    # source: 2504.00077, eqn 2.15
    # MB equation of state is ideal-gas exactly: P = n T
    return n_i * Th



def safe_exp(logx):
    """exp(logx) with an underflow clamp to 0 ."""
    return np.exp(np.maximum(logx, LOG_TINY))

def ln_neq(m, g, T):
    """
    ln n_eq = ln[g m^2 T /(2π^2)] + ln K2(m/T)
            = ln[g m^2 T /(2π^2)] + (-z + ln kve(2, z)),  z = m/T
    """
    z = m / T
    return (np.log(g) + 2*np.log(m) + np.log(T) - np.log(2*np.pi**2)
            - z + np.log(kve(2, z)))


def _R(z):
    #ratio of bessels K1(z)/K2(z)
    return kve(1, z) / kve(2, z)  # stable, exp factors cancel

def _dR_dz(z, rel_step=1e-6):
    #z = max(z, 1e-12)
    dz = np.maximum(rel_step * z, 1e-6)
    Rp = kve(1, z+dz)/kve(2, z+dz)
    Rm = kve(1, z-dz)/kve(2, z-dz)
    return (Rp - Rm) / (2.0*dz)

def dTh_dt_rel(nchi, nA, Th, mchi, mA, H, rhoh, Ph, Qh, dnchi_dt, dnA_dt):
    zc = mchi /np.maximum(Th, 1e-300)
    zA = mA   / np.maximum(Th, 1e-300)

    Rchi = _R(zc)
    RA = _R(zA)
    dRchi_dz = _dR_dz(zc)
    dRA_dz = _dR_dz(zA)

    # h_i
    h_chi = mchi * Rchi + 3.0 * Th
    h_A   = mA   * RA   + 3.0 * Th

    # C_h = sum n_i [ 3 - z_i^2 dR/dz ] HEAT CAPACITY at constant composition
    Ch = nchi * (3.0 - zc*zc * dRchi_dz) + nA * (3.0 - zA*zA * dRA_dz)
    Ch = max(Ch, 1e-300)

    numer = Qh - (h_chi * dnchi_dt + h_A * dnA_dt) - 3.0 * H * (rhoh + Ph)
    return numer / Ch



def collisions(params, nchi, nA, T, Th):
    mchi = params["mchi"]; mA = params["mA"]
    gchi = params["gchi"]; gA = params["gA"]
    sv_xxee = params["sv_xxee"]; sv_xxAA = params["sv_xxAA"]
    gamma_Aee = params["gamma_Aee"]

    # log equilibrium densities (SM at T, HS at Th)
    ln_nchi_eq_SM = ln_neq(mchi, gchi, T)
    ln_nA_eq_SM   = ln_neq(mA,   gA,   T)
    ln_nchi_eq_Th = ln_neq(mchi, gchi, Th)
    ln_nA_eq_Th   = ln_neq(mA,   gA,   Th)

    # Detailed-balance factor: (nchi_eq_Th^2 / nA_eq_Th^2) * nA^2  in log-space
    # ln balance = 2(ln nx_eq^Th − ln nA_eq^Th) + 2 ln nA
    ln_balance = 2.0*(ln_nchi_eq_Th - ln_nA_eq_Th) + 2.0*np.log(max(nA, VAL_FLOOR))
    balance = safe_exp(ln_balance)  # safely 0 if hugely negative

    # Equilibrium values (SM), clamped
    nchi_eq_SM = safe_exp(ln_nchi_eq_SM)
    nA_eq_SM   = safe_exp(ln_nA_eq_SM)

    # Stable Lorentz factor gamma = <m/E> = K1/K2;  scaled kve ratio = kv(1,z)/kv(2,z)
    zA = mA / np.maximum(Th, VAL_FLOOR)
    lorentz_factor_HS = kve(1, zA) / kve(2, zA)

    zA_SM = mA / np.maximum(T, VAL_FLOOR)
    lorentz_factor = kve(1, zA_SM) / np.maximum(kve(2, zA_SM), VAL_FLOOR)

    # Number collision terms
    C_chi =  - sv_xxAA*(nchi**2 - balance) -0.5*sv_xxee*(nchi**2 - nchi_eq_SM**2)
    C_A   =    sv_xxAA*(nchi**2 - balance) - gamma_Aee *(nA*lorentz_factor_HS - nA_eq_SM*lorentz_factor)

    # Energy transfer eqns
    Q_ann = -0.5 * mchi * sv_xxee * (nchi**2 - nchi_eq_SM**2)
    Q_dec = - mA   * gamma_Aee * (nA - nA_eq_SM)  #no lorentz factor here
    Q_h   = Q_ann + Q_dec

    return C_chi, C_A, Q_h


def H_of_T(T, rho_h=0.0, t_dep = False, include_hs = True):
    """Hubble from SM  + HS """
    ## typically by freezeout, all radiation energy has been deposited into non-rel species
    ## technically more correct to include
    rho_SM = (np.pi**2/30.0) * gstar_SM(T, t_dep) * T**4
    if include_hs:
        rho_tot = rho_SM + rho_h
    else:
        rho_tot = rho_SM
    return np.sqrt((8.0*np.pi/3.0) * rho_tot) / MPL



# ----------------------------
# x-domain RHS: d/dx = (1/(dx/dt)) d/dt
# ----------------------------
def rhs_logx(x, u, params):
    """
    log-space RHS:
      u = [ln n_chi, ln n_Ap, ln Th]. returns d/dx of those logs
    """
    ln_nchi, ln_nA, ln_Th  = u
    nchi = np.exp(ln_nchi)
    nA   = np.exp(ln_nA)
    Th   = np.exp(ln_Th)

    mchi = params["mchi"];  mA = params["mA"]
    t_dep = params.get("t_dep", True)
    include_h = params.get("include_hs_in_H", True)
    T    = mchi / x # SM temperature ('convenient monotonic variable')

    # HS energy density and pressure
    rhoh = rho_i_exact(nchi, mchi, Th) + rho_i_exact(nA, mA, Th)
    Ph = P_i_exact(nchi, Th) + P_i_exact(nA, Th)

    # Hubble and x-dot
    H    = H_of_T(T, rhoh if include_h else 0.0, t_dep = t_dep)
    xdot = dx_dt(T, H, m = mchi, t_dep = t_dep)

    # collisions
    C_chi, C_A, Qh = collisions(params, nchi, nA, T, Th)

    # Time derivatives for number rxns
    dnchi_dt = -3.0 * H * nchi + C_chi
    dnA_dt   = -3.0 * H * nA   + C_A
    dTh_dt = dTh_dt_rel(nchi, nA, Th, mchi, mA, H, rhoh, Ph, Qh, dnchi_dt, dnA_dt)

    dnchi_dx = dnchi_dt /xdot
    dnA_dx   = dnA_dt   /xdot
    dTh_dx   = dTh_dt   /xdot

    # Return log-derivatives
    dln_nchi_dx = dnchi_dx / nchi
    dln_nA_dx   = dnA_dx / nA
    dln_Th_dx   = dTh_dx / Th

    return np.array([dln_nchi_dx, dln_nA_dx, dln_Th_dx])


def neq_stable(m, g, T):
    T = np.maximum(T, VAL_FLOOR)
    z = m / T
    # ln n_eq = ln[g m^2 T /(2π^2)] - z + ln kve(2,z)
    ln = (np.log(g) + 2*np.log(m) + np.log(T) - np.log(2*np.pi**2)
          - z + np.log(kve(2, z)))
    return safe_exp(ln)

def compute_diagnostics(xs, sol_y, params):
    ln_nchi, ln_nA, ln_Th = sol_y
    nchi = np.exp(ln_nchi); nA = np.exp(ln_nA); Th = np.exp(ln_Th)

    mchi, mA   = params["mchi"], params["mA"]
    gchi = params["gchi"]; gA = params["gA"]
    sv_xxAA    = params["sv_xxAA"]; sv_xxee = params["sv_xxee"]
    gammaA     = params["gamma_Aee"]
    t_dep      = params.get("t_dep", True)
    include_h  = params.get("include_hs_in_H", True)


    T   = mchi / xs
    # HS thermodynamics
    rhoh = rho_i_exact(nchi, mchi, Th) + rho_i_exact(nA, mA, Th)
    H    = H_of_T(T, rhoh if include_h else 0.0, t_dep=t_dep)

    # Equilibria at HS temperature
   #nxeq_Th = neq(mchi, gchi, Th)
    #naeq_Th = neq(mA,   gA,   Th)

    nxeq_Th = neq_stable(mchi, gchi, Th)
    naeq_Th = neq_stable(mA,   gA,   Th)

    # Per-particle reaction rates (no cancellation)
    Gamma_xAA_over_H = (sv_xxAA * nchi) / H
    Gamma_xSM_over_H = (sv_xxee * nchi) / H
    zA = mA / np.maximum(Th, VAL_FLOOR)
    lorentz = kve(1, zA) / np.maximum(kve(2, zA), VAL_FLOOR)
    Gamma_Adec_over_H = (gammaA * lorentz) / H
    Gamma_Adec_over_H_no_lorentz = (gammaA) / H

    # collision terms + energy transfer
    Cchi = np.empty_like(xs); CA = np.empty_like(xs); Qh = np.empty_like(xs)
    for i in range(xs.size):
        Cchi[i], CA[i], Qh[i] = collisions(params, nchi[i], nA[i], T[i], Th[i])

    # Compete with Hubble dilution
    C_over_3Hn_chi = np.abs(Cchi) / np.maximum(3*H*nchi, VAL_FLOOR)
    C_over_3Hn_A   = np.abs(CA)   / np.maximum(3*H*nA,   VAL_FLOOR)

    # energy exchange strength
    Q_over_Hrho = np.abs(Qh) / np.maximum(H * rhoh, VAL_FLOOR)

    dep_chi = np.maximum(nchi / np.maximum(nxeq_Th, VAL_FLOOR), VAL_FLOOR)
    dep_A   = np.maximum(nA   / np.maximum(naeq_Th, VAL_FLOOR), VAL_FLOOR)

    Gamma_chem_over_H = np.abs(Cchi) / (H * np.maximum(np.abs(nchi - nxeq_Th), VAL_FLOOR))

    return {
        "x": xs, "T": T, "Th": Th, "H": H,
        "Gamma_xAA_over_H": Gamma_xAA_over_H,
        "Gamma_xSM_over_H": Gamma_xSM_over_H,
        "Gamma_Adec_over_H": Gamma_Adec_over_H,
        "Gamma_Adec_over_H_no_lorentz": Gamma_Adec_over_H_no_lorentz,
        "Gamma_chem_over_H": Gamma_chem_over_H,

        "C_over_3Hn_chi": C_over_3Hn_chi,
        "C_over_3Hn_A": C_over_3Hn_A,
        "Q_over_Hrho": Q_over_Hrho,
        "dep_chi": dep_chi,
        "dep_A": dep_A,

        "nchi_over_nchieq": nchi / np.maximum(nxeq_Th, VAL_FLOOR),
        "nA_over_nAeq": nA / np.maximum(naeq_Th, VAL_FLOOR),
        "params": params,
    }


def evolve(params, x_initial, x_final, y0, xs, log_space = False):
    if log_space:
        u0 = np.log(y0)
        sol = solve_ivp(rhs_logx, (x_initial, x_final), u0, args = (params,), t_eval=xs, method="Radau", rtol=1e-7, atol=1e-14) #, max_step=0.5)
        diag = compute_diagnostics(sol.t, sol.y, params)
    #else:
        #sol = solve_ivp(rhs_x, (x_initial, x_final), y0, args = (params,), t_eval=xs, method="Radau", rtol=1e-7, atol=1e-12, max_step=0.5)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    x_arr = sol.t
    if log_space:
        soly = np.exp(sol.y)
    else:
        soly = sol.y
    return {"sol": soly, "x_arr": x_arr, "diag": diag}


def s_SM(T, t_dep = False):
    """SM entropy density"""
    return (2.0*np.pi**2/45.0) * gstars_SM(T, t_dep) * T**3

def get_relic_abundance(Y, m):
    return 2.742e8 * m * Y

def Y(n, T): return n / s_SM(T)
