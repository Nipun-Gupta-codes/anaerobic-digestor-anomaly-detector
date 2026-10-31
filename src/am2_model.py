"""
AM2 Two-Step Acidogenesis-Methanogenesis Kinetic Model
======================================================
Reference: Bernard, O., Hadj-Sadok, Z., Dochain, D., Genovesi, A., & Steyer, J.-P. (2001).
           Dynamical model development and parameter identification for an anaerobic
           wastewater treatment process.  Biotechnology & Bioengineering, 75(4), 424-438.
           https://doi.org/10.1002/bit.10036

State variables (all concentrations in g/L or mol/L as specified):
  S1  – soluble organic substrate (g COD/L or g VS/L)
  X1  – acidogenic (acidogens) biomass concentration (g VS/L)
  S2  – volatile fatty acids / total VFA as acetate equivalent (mmol/L)
  X2  – methanogenic (methanogens) biomass concentration (g VS/L)

Gaseous output:
  q_CH4 – methane volumetric production rate (L·CH4 / (L·reactor·day))

Governing equations (Bernard et al., 2001, Table 1 & Eq. 4–14):
  dS1/dt = D*(S1_in - S1)  - k1*mu1(S1)*X1
  dX1/dt = (mu1(S1) - alpha*D)*X1
  dS2/dt = D*(S2_in - S2)  + k2*mu1(S1)*X1 - k3*mu2(S2)*X2
  dX2/dt = (mu2(S2) - alpha*D)*X2

where:
  D     = dilution rate (1/day)  = feed flow / working volume
  alpha = fraction of biomass leaving with effluent (1 for CSTR, 0 for retained sludge)
  k1    = yield coefficient for S1 consumption by X1 (g COD/g VS)
  k2    = VFA yield per unit acidogenic growth (mmol VFA/g VS)
  k3    = VFA consumption yield by X2 (mmol VFA/g VS)
  mu1   = Monod growth rate for acidogens (1/day)
  mu2   = Haldane growth rate for methanogens with VFA inhibition (1/day)

Growth kinetics:
  mu1(S1) = mu1_max * S1 / (K_S1 + S1)            [Monod]
  mu2(S2) = mu2_max * S2 / (K_S2 + S2 + S2^2/K_I) [Haldane / Andrews – VFA inhibition]

Methane production rate (ideal gas at 35°C, 1 atm):
  q_CH4 = k6 * mu2(S2) * X2     where k6 = mmol CH4/g VS·day -> molar conversion

Literature parameter values (Bernard et al., 2001, Table 2 – "nominal"):
  mu1_max = 1.20  1/day
  mu2_max = 0.74  1/day
  K_S1    = 7.10  g COD/L
  K_S2    = 9.28  mmol/L
  K_I     = 256   mmol/L     (VFA inhibition constant)
  k1      = 42.14 g COD/g VS (= Y1^-1 if Y1~0.0237)
  k2      = 116.5 mmol/g VS
  k3      = 268.0 mmol/g VS
  k6      = 453.0 mmol CH4/g VS
  D       = operating variable (1/day)
  alpha   = 1.0   (CSTR, no biomass retention)

pH is calculated from VFA / total-alkalinity using a simplified carbonate-buffer model.
This is a literature simulation, NOT calibrated against any specific plant.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Default parameter set (Bernard et al., 2001)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    mu1_max=1.20,    # max acidogen growth rate (1/day)
    mu2_max=0.74,    # max methanogen growth rate (1/day)
    K_S1=7.10,       # half-saturation constant for S1 (g COD/L)
    K_S2=9.28,       # half-saturation constant for S2 (mmol/L)
    K_I=256.0,       # Haldane inhibition constant for VFA (mmol/L)
    k1=42.14,        # S1 consumption stoichiometry (g COD/g VS)
    k2=116.5,        # VFA yield from acidogenesis (mmol/g VS)
    k3=268.0,        # VFA consumption stoichiometry (mmol/g VS)
    k6=453.0,        # CH4 yield from methanogenesis (mmol CH4/g VS)
    alpha=1.0,       # effluent fraction of biomass (1 = CSTR)
    # decay coefficients (not in original Bernard model; added conservatively)
    b1=0.01,         # acidogen decay (1/day)  – approximation
    b2=0.01,         # methanogen decay (1/day) – approximation
)

# Molar volume of CH4 at 35°C (308 K), 1 atm  [L/mol]
T_K = 308.0
R_GAS = 0.08206  # L·atm/(mol·K)
V_CH4_MOLAR = R_GAS * T_K  # ≈ 25.3 L/mol

# ---------------------------------------------------------------------------
# Alkalinity / pH model
# Simplified: total alkalinity (mg/L as CaCO3) is approximated; pH computed
# from buffering equilibrium.  This is a qualitative indicator only.
# ---------------------------------------------------------------------------
CaCO3_MW = 100.09   # g/mol
CH3COOH_MW = 60.052  # g/mol as acetic acid
KA2 = 4.73e-7        # Ka2 of CO2/HCO3- at 35°C (mol/L)
KH = 2.74e-2         # Henry's constant for CO2 at 35°C (mol/L/atm)
P_CO2 = 0.40         # partial pressure CO2 in biogas headspace (atm)

def vfa_mmol_to_mgl(s2_mmol):
    """Convert VFA (as acetate, mmol/L) to mg/L as acetic acid equivalent."""
    return s2_mmol * CH3COOH_MW

def estimate_alkalinity_and_pH(S2_mmol: float, alk_in_mmol: float) -> tuple[float, float]:
    """
    Estimate total alkalinity (mg/L as CaCO3) and pH from VFA and influent alkalinity.
    S2_mmol  : current VFA in mmol/L
    alk_in_mmol : influent alkalinity in mmol/L (as HCO3-)
    Returns: (alkalinity mg/L CaCO3, pH)
    """
    # Consumed by VFA (1:1 proton exchange with bicarbonate)
    alk_mmol = max(alk_in_mmol - S2_mmol, 0.5)       # never go below ~25 mg/L CaCO3
    alk_mgl = alk_mmol * CaCO3_MW / 2.0              # mmol HCO3- to mg/L CaCO3 (equiv. weight 50)
    # Carbonate equilibrium: [H+] from CO2 / HCO3- ratio
    HCO3 = alk_mmol * 1e-3                            # mol/L
    H2CO3 = KH * P_CO2                                # dissolved CO2 (mol/L)
    H_conc = KA2 * H2CO3 / max(HCO3, 1e-8)
    pH = -np.log10(max(H_conc, 1e-14))
    pH = np.clip(pH, 4.5, 9.0)
    return alk_mgl, pH


# ---------------------------------------------------------------------------
# Kinetic functions
# ---------------------------------------------------------------------------
def mu1(S1: float, p: dict) -> float:
    """Monod growth rate for acidogens."""
    return p['mu1_max'] * S1 / (p['K_S1'] + S1)

def mu2(S2: float, p: dict) -> float:
    """Haldane (Andrew) growth rate for methanogens; VFA self-inhibition."""
    return p['mu2_max'] * S2 / (p['K_S2'] + S2 + S2**2 / p['K_I'])


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------
def am2_ode(t: float, y: np.ndarray, p: dict,
            D_func: Callable, S1_in_func: Callable, S2_in_func: Callable) -> np.ndarray:
    """
    AM2 system right-hand side.

    Parameters
    ----------
    t         : current time (days)
    y         : state [S1, X1, S2, X2]
    p         : parameter dict
    D_func    : callable D(t) -> dilution rate (1/day)
    S1_in_func: callable S1_in(t) -> influent substrate (g COD/L)
    S2_in_func: callable S2_in(t) -> influent VFA (mmol/L), usually ~0
    """
    S1, X1, S2, X2 = np.maximum(y, 0.0)   # physical bound
    D     = D_func(t)
    S1_in = S1_in_func(t)
    S2_in = S2_in_func(t)
    m1 = mu1(S1, p)
    m2 = mu2(S2, p)
    dS1 = D * (S1_in - S1) - p['k1'] * m1 * X1
    dX1 = (m1 - p['alpha'] * D - p['b1']) * X1
    dS2 = D * (S2_in - S2) + p['k2'] * m1 * X1 - p['k3'] * m2 * X2
    dX2 = (m2 - p['alpha'] * D - p['b2']) * X2
    return np.array([dS1, dX1, dS2, dX2])


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
@dataclass
class AM2Result:
    t: np.ndarray          # time (days)
    S1: np.ndarray         # substrate (g COD/L)
    X1: np.ndarray         # acidogen biomass (g VS/L)
    S2: np.ndarray         # VFA (mmol/L)
    X2: np.ndarray         # methanogen biomass (g VS/L)
    q_CH4_mmol: np.ndarray # CH4 production rate (mmol/L/day)
    q_CH4_Ld: np.ndarray   # CH4 production rate (L CH4/L reactor/day)
    vfa_mgl: np.ndarray    # VFA as acetic acid (mg/L)
    alkalinity_mgl: np.ndarray
    pH: np.ndarray
    D: np.ndarray          # dilution rate used at each t
    label: np.ndarray      # integer scenario label per time point
    label_name: np.ndarray # string label name per time point


def solve_am2(
    t_span: tuple[float, float],
    y0: list[float],
    D_func: Callable,
    S1_in_func: Callable,
    S2_in_func: Callable = None,
    alk_in_mmol: float = 25.0,
    params: dict = None,
    label_func: Callable = None,
    label_map: dict = None,
    t_eval: np.ndarray = None,
    max_step: float = 0.1,
) -> AM2Result:
    """
    Solve the AM2 model over t_span using scipy solve_ivp (LSODA method).

    Parameters
    ----------
    t_span      : (t_start, t_end) in days
    y0          : [S1_0, X1_0, S2_0, X2_0] initial conditions
    D_func      : D(t) -> dilution rate (1/day)
    S1_in_func  : S1_in(t) -> influent substrate concentration (g COD/L)
    S2_in_func  : S2_in(t) -> influent VFA (mmol/L); defaults to zero
    alk_in_mmol : influent alkalinity as HCO3- (mmol/L); default 25 mmol/L (~1250 mg/L CaCO3)
    params      : override any DEFAULT_PARAMS keys
    label_func  : t -> integer label; defaults to 0 (Normal)
    label_map   : {int: str} mapping label ints to names
    t_eval      : evaluation time points; if None, daily points are used
    max_step    : max integration step (days)

    Returns
    -------
    AM2Result dataclass
    """
    if params is None:
        params = {}
    p = {**DEFAULT_PARAMS, **params}

    if S2_in_func is None:
        S2_in_func = lambda t: 0.0
    if label_func is None:
        label_func = lambda t: 0
    if label_map is None:
        label_map = {0: 'Normal', 1: 'Overloading', 2: 'Early Acidification',
                     3: 'High VFA', 4: 'Severe Souring', 5: 'Recovery'}
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1] + 0.5, 1.0)  # daily

    # Try Radau first (stiff-safe), fall back to LSODA
    for method in ('Radau', 'LSODA', 'BDF'):
        sol = solve_ivp(
            fun=lambda t, y: am2_ode(t, y, p, D_func, S1_in_func, S2_in_func),
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-6, atol=1e-9,
        )
        if sol.success:
            break
    if not sol.success:
        raise RuntimeError(f"ODE solver failed ({method}): {sol.message}")

    S1, X1, S2, X2 = np.maximum(sol.y, 0.0)
    q_CH4_mmol = p['k6'] * np.vectorize(lambda s2: mu2(s2, p))(S2) * X2
    q_CH4_Ld   = q_CH4_mmol * V_CH4_MOLAR / 1000.0  # mmol -> mol -> L (gas)

    vfa_mgl  = vfa_mmol_to_mgl(S2)
    alk_arr  = np.empty_like(S2)
    pH_arr   = np.empty_like(S2)
    for i, s2 in enumerate(S2):
        alk_arr[i], pH_arr[i] = estimate_alkalinity_and_pH(s2, alk_in_mmol)

    D_arr     = np.array([D_func(tt) for tt in sol.t])
    label_arr = np.array([label_func(tt) for tt in sol.t], dtype=int)
    lname_arr = np.array([label_map.get(int(l), 'Unknown') for l in label_arr])

    return AM2Result(
        t=sol.t, S1=S1, X1=X1, S2=S2, X2=X2,
        q_CH4_mmol=q_CH4_mmol, q_CH4_Ld=q_CH4_Ld,
        vfa_mgl=vfa_mgl, alkalinity_mgl=alk_arr, pH=pH_arr,
        D=D_arr, label=label_arr, label_name=lname_arr,
    )
