#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLARE  -  Feedback Loop Activated by Redox Events
Reproduces every figure in the FLARE document (Figures 1-10).

Part of BOREAS  |  Chassis: Lactiplantibacillus plantarum ATCC 8014
Sensor: PerR / Spx / P_trxB

------------------------------------------------------------------------------
HOW TO USE IN GOOGLE COLAB
------------------------------------------------------------------------------
1. Upload this file, or paste each SECTION into its own cell.
2. Run SECTION 0 first (imports, parameters, model).  Everything else depends
   on it.
3. Each figure lives in its own function, fig01() ... fig10().  Call the one
   you want, or run main() at the bottom for all ten.
4. Figures are saved as PNG at 300 dpi into ./flare_figures/ and also shown
   inline.

Only numpy, scipy and matplotlib are required - all present in Colab by
default.  No other files are needed; this script is self-contained.

------------------------------------------------------------------------------
WHAT SHOULD REPRODUCE
------------------------------------------------------------------------------
  midpoint (fitted)            48.01 uM applied H2O2
  effective Hill coefficient   2.12
  P_trxB at 250 uM             0.787
  P_trxB OFF / saturating      0.010 / 0.832
  induction ratio              83.2x
  time to half-maximal output  0.67 h
  output half-decay            0.19 h
  PerR recovery half-time      1.50 h
Run verify() to check these against the document.

------------------------------------------------------------------------------
FIGURE INDEX  (one function per figure in the document, nothing extra)
------------------------------------------------------------------------------
  fig01()   S4.1  steady-state H2O2 and P_trxB vs lesion production, both K_m
                  scenarios; 63.2% (A) and 15.5% (B) midpoint crossings
  fig02()   S4.1  cell density moves the midpoint boundary through Vmax = q_cell N
                  -> 8.5e7 (A) and 18.9e7 CFU/mL (B)
  fig03()   S4.2  Gate 1 depends only on the product k_PerR,eff = phi_Fe * k_Fe
  fig04()   S4.4  the two gates separately and combined -> 48.05 uM, n_eff 2.12
  fig05()   S4.5  k_ox sweep; fitted midpoint moves from about 76 to 24 uM
  fig06()   S4.6  onset/withdrawal kinetics, and decay at full dose and at
                  1% residual density (4.1 / 10.5 h and 17.2 / 43.8 d)
  fig07()   S4.7  coupled lesion-circuit runs per clearance capacity, plus the
                  section 4.7 H_ss and days-above-midpoint table
  fig08()   S6    local +/-10% sensitivity on two different endpoints
  fig09_10() S6   16-parameter Latin-hypercube global analysis and the rank
                  correlations with the fitted midpoint

Every figure prints the document's own numbers next to the computed ones, so a
mismatch is visible without opening the document.

NOTE ON FIGURE 9-10 (global sensitivity): the document used 4000 Latin-
hypercube samples per ClpXP version, which takes roughly 10-25 min in Colab.
N_LHS below is set to 600 so the notebook runs quickly.  Set N_LHS = 4000 to
reproduce the published statistics exactly.
==============================================================================
"""

# =============================================================================
# SECTION 0  -  IMPORTS, PARAMETERS, MODEL
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, brentq
from scipy.stats import qmc, spearmanr

OUTDIR = "flare_figures"
os.makedirs(OUTDIR, exist_ok=True)

# ---- FROST/BOREAS brand palette -------------------------------------------
NAVY = "#005389"   # primary
MID  = "#5498CA"   # secondary
ICE  = "#9FD7ED"   # fill / highlight
TAN  = "#D39B5E"   # contrast
INK  = "#1a2733"   # text
GRID = "#dce7ee"
RED  = "#9c3b28"   # thresholds
GREY = "#9aa8b2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ---- Table 5 parameters ----------------------------------------------------
P = dict(
    # --- input coupling (Eq 3) ---
    f_in      = 0.107,    # EST  [1]   applied -> intracellular fraction
    chi_OX    = 0.40,     # EST  [2]   eq-uM oxidant per uM applied H2O2

    # --- Gate 1: PerR (Eqs 4-5) ---
    PerR_tot  = 6.0,      # EST  [5]   uM
    k_Fe      = 12.5,     # DER  [4]   uM^-1 h^-1
    phi_Fe    = 0.05,     # EST  [3,11,12,13]
    delta_Per = 0.462,    # EST  [3,17] h^-1
    K_perR    = 1.159,    # DER  [5]   uM   (two-point fit)
    n_perR    = 1.212,    # DER  [5]

    # --- ClpXP gate (Eq 6) ---
    k_clp     = 6.0,      # EST  [6]   h^-1
    min_clp   = 0.097,    # DER  [6]   upper bound
    K_clp     = 1.5,      # EST  [6]   uM
    n_clp     = 2,        # EST  [6]

    # --- Spx pool (Eqs 7-9) ---
    k_syn     = 3.0,      # EST        uM h^-1
    delta_spx = 0.35,     # EST        h^-1
    k_ox      = 0.060,    # EST  [7]   (eq-uM)^-1 h^-1
    k_trx     = 1.20,     # EST  [7]   uM^-1 h^-1

    # --- reducing capacity (Eq 11) ---
    Trx_bas   = 0.50,     # EST  [7]   uM
    Trx_max   = 0.90,     # DER  [18]  uM  (= 1.8 x Trx_bas)
    k_rel     = 2.0,      # EST        h^-1

    # --- output promoter (Eq 10) ---
    P_basal   = 0.010,    # EST
    K_out     = 0.60,     # EST  [9]   uM
    n_out     = 1,        # EST  [8]
)

# ---- clearance parameters (Eqs 1-2) ---------------------------------------
NPX_SA   = 33.5        # LIT [21]  nmol min^-1 mg protein^-1
M_PROT   = 0.500       # EST [21]  pg protein per CFU  (implied)
Q_CELL   = 1.005       # DER [21]  fmol h^-1 cell^-1
N_DESIGN = 6e7         #           CFU mL^-1, design dose
VMAX     = 60.3        # DER [21]  uM h^-1 at N_DESIGN
KM_A     = 28.0        # LIT [22]  uM   Scenario A, applied basis
KM_B     = KM_A / P["f_in"]   #    uM   Scenario B = 261.7 uM
H_LESION = 250.0       #           uM   assumed lesional steady state
H_MID    = 48.0        #           uM   sensor midpoint


def hill(x, K, n):
    """Hill function; x, K > 0."""
    x = np.maximum(x, 0.0)
    return x**n / (K**n + x**n)


def rhs(t, y, I_of_t, p):
    """
    Eqs 3-11.  State vector y = [PerR, Spx_red, Spx_ox, Trx].

    Eq 3   I_int = f_in * I ;  OX = chi_OX * I
    Eq 4   dPerR/dt = delta_Per*(PerR_tot - PerR) - phi_Fe*k_Fe*I_int*PerR
    Eq 5   P_spx   = 1 / (1 + (PerR/K_perR)^n_perR)
    Eq 6   g_clp   = 1 - (1-min_clp) * I_int^n / (K_clp^n + I_int^n)
    Eq 7   v_ox    = k_ox * OX * Spx_red ;  v_re = k_trx * Trx * Spx_ox
    Eq 8   dSpx_red/dt = k_syn*P_spx - delta_tot*Spx_red - v_ox + v_re
    Eq 9   dSpx_ox /dt =              - delta_tot*Spx_ox  + v_ox - v_re
    Eq 10  P_trxB  = P_basal + (1-P_basal) * Spx_ox^n_out/(K_out^n_out + Spx_ox^n_out)
    Eq 11  dTrx/dt = k_rel*(Trx_bas + (Trx_max-Trx_bas)*P_trxB - Trx)
    """
    PerR, S_red, S_ox, Trx = y
    I = I_of_t(t)
    I_int = p["f_in"] * I                      # Eq 3
    OX    = p["chi_OX"] * I                    # Eq 3

    P_spx = 1.0 / (1.0 + (max(PerR, 0.0) / p["K_perR"])**p["n_perR"])   # Eq 5
    g_clp = 1.0 - (1.0 - p["min_clp"]) * hill(I_int, p["K_clp"], p["n_clp"])  # Eq 6
    d_tot = p["k_clp"] * g_clp + p["delta_spx"]

    v_ox = p["k_ox"]  * OX  * S_red            # Eq 7
    v_re = p["k_trx"] * Trx * S_ox             # Eq 7

    dPerR  = p["delta_Per"] * (p["PerR_tot"] - PerR) \
             - p["phi_Fe"] * p["k_Fe"] * I_int * PerR          # Eq 4
    dS_red = p["k_syn"] * P_spx - d_tot * S_red - v_ox + v_re  # Eq 8
    dS_ox  = -d_tot * S_ox + v_ox - v_re                       # Eq 9

    P_trxB = p["P_basal"] + (1 - p["P_basal"]) * hill(S_ox, p["K_out"], p["n_out"])  # Eq 10
    dTrx   = p["k_rel"] * (p["Trx_bas"]
                           + (p["Trx_max"] - p["Trx_bas"]) * P_trxB - Trx)           # Eq 11
    return [dPerR, dS_red, dS_ox, dTrx]


def steady_state(I, p=None, t_end=400.0):
    """Integrate to steady state at constant applied H2O2 = I (uM)."""
    p = p or P
    y0 = [p["PerR_tot"], 0.0, 0.0, p["Trx_bas"]]
    s = solve_ivp(rhs, (0, t_end), y0, args=(lambda t: I, p),
                  method="LSODA", rtol=1e-9, atol=1e-12)
    PerR, S_red, S_ox, Trx = s.y[:, -1]
    P_trxB = p["P_basal"] + (1 - p["P_basal"]) * hill(S_ox, p["K_out"], p["n_out"])
    return dict(I=I, PerR=PerR, Spx_red=S_red, Spx_ox=S_ox,
                Spx_tot=S_red + S_ox,
                f_ox=S_ox / max(S_red + S_ox, 1e-30),
                Trx=Trx, P_trxB=P_trxB)


def simulate(I_of_t, p=None, t_end=48.0, n=4000, breakpoints=()):
    """Time course with a (possibly discontinuous) input I(t)."""
    p = p or P
    y0 = np.array([p["PerR_tot"], 0.0, 0.0, p["Trx_bas"]])
    edges = sorted(set([0.0] + list(breakpoints) + [t_end]))
    T, Y = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        pts = max(int(n * (b - a) / t_end), 50)
        s = solve_ivp(rhs, (a, b), y0, args=(I_of_t, p), method="LSODA",
                      t_eval=np.linspace(a, b, pts), rtol=1e-9, atol=1e-12)
        T.append(s.t); Y.append(s.y); y0 = s.y[:, -1]
    T = np.concatenate(T); Y = np.hstack(Y)
    P_trxB = p["P_basal"] + (1 - p["P_basal"]) * hill(Y[2], p["K_out"], p["n_out"])
    return T, Y, P_trxB


def dose_response(p=None, lo=-2, hi=3.2, n=140):
    """Steady-state P_trxB across applied H2O2."""
    xs = np.logspace(lo, hi, n)
    return xs, np.array([steady_state(x, p)["P_trxB"] for x in xs])


def fit_hill(x, y):
    """Fit a 2-parameter Hill curve; returns (K50, n_eff)."""
    y0, y1 = y.min(), y.max()
    f = lambda xx, K, n: y0 + (y1 - y0) * xx**n / (K**n + xx**n)
    try:
        popt, _ = curve_fit(f, x, y, p0=[np.median(x), 2.0], maxfev=40000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


# ---- clearance helpers (Eqs 1-2) ------------------------------------------
def H_ss(P_les, Km, Vmax=VMAX):
    """Eq 2 rearranged: steady-state H for production P_les."""
    if P_les >= Vmax:
        return np.inf
    return Km * P_les / (Vmax - P_les)


def r_star(Km, H_target=H_MID):
    """Fraction of Vmax at which H_ss equals H_target."""
    return H_target / (Km + H_target)


def P_les_for(H_target, Km, Vmax=VMAX):
    """Production rate that holds H at H_target."""
    return Vmax * H_target / (Km + H_target)


def save(fig, name):
    path = os.path.join(OUTDIR, name + ".png")
    fig.savefig(path)
    print("  saved ->", path)


# =============================================================================
# VERIFICATION  -  check the model reproduces the document's headline numbers
# =============================================================================
def verify():
    print("=" * 74)
    print("FLARE verification  (model output vs document)")
    print("=" * 74)
    xs, ys = dose_response(n=220)
    K, n = fit_hill(xs, ys)
    sp = np.array([steady_state(x)["Spx_tot"] for x in xs])
    Ka, na = fit_hill(xs, sp)

    off = steady_state(0.01)["P_trxB"]
    sat = steady_state(1000.0)["P_trxB"]
    les = steady_state(H_LESION)["P_trxB"]

    # onset / decay dynamics
    I_on = lambda t: 100.0 if t >= 2.0 else 0.01
    T, Y, Pt = simulate(I_on, t_end=30, n=6000, breakpoints=[2.0])
    m = T >= 2.0
    tgt = Pt[m][0] + 0.5 * (Pt[m].max() - Pt[m][0])
    t_rise = T[m][np.argmax(Pt[m] >= tgt)] - 2.0

    I_off = lambda t: 100.0 if t < 12.0 else 0.01
    T2, Y2, Pt2 = simulate(I_off, t_end=40, n=8000, breakpoints=[12.0])
    m2 = T2 >= 12.0
    half = Pt2[m2][0] - 0.5 * (Pt2[m2][0] - Pt2[m2].min())
    t_dec = T2[m2][np.argmax(Pt2[m2] <= half)] - 12.0
    # PerR recovery is resynthesis-limited: the deficit halves with time
    # constant ln2 / delta_Per  (Eq 4, the irreversible-oxidation term is off)
    t_perr = np.log(2) / P["delta_Per"]

    rows = [
        ("midpoint, fitted (uM)",        K,    48.01),
        ("effective Hill n",             n,     2.12),
        ("abundance-gate K (uM)",        Ka,   58.9),
        ("abundance-gate n",             na,    1.68),
        ("P_trxB at 250 uM",             les,   0.787),
        ("P_trxB OFF",                   off,   0.010),
        ("P_trxB saturating (1 mM)",     sat,   0.832),
        ("induction ratio",              sat / off, 83.2),
        ("time to half-max output (h)",  t_rise, 0.67),
        ("output half-decay (h)",        t_dec,  0.19),
        ("PerR recovery half-time (h)",  t_perr, 1.50),
    ]
    for lab, got, doc in rows:
        tol = max(0.03 * abs(doc), 0.02)
        flag = "ok " if abs(got - doc) <= tol else "CHECK"
        print(f"  {lab:32s} {got:9.3f}   doc {doc:8.3f}   {flag}")

    print()
    print("  clearance chain")
    print(f"    q_cell  = Vmax/N        = {VMAX*1e6/N_DESIGN:8.4f} fmol/h/cell   doc 1.005")
    print(f"    m_prot  = q_cell/SA     = {Q_CELL/(NPX_SA*60*1e6)*1e9:8.4f} pg/CFU        doc 0.500")
    print(f"    Km_B    = Km_A/f_in     = {KM_B:8.1f} uM            doc 261.7")
    print(f"    r* (A)  = 48/(28+48)    = {100*r_star(KM_A):8.1f} %             doc 63.2")
    print(f"    r* (B)  = 48/(261.7+48) = {100*r_star(KM_B):8.1f} %             doc 15.5")
    print("=" * 74)


# =============================================================================
# FIGURE 1  -  Section 4.1
# Compartment choice changes how lesion production maps to H2O2
# =============================================================================
def fig01():
    print("Figure 1 ...")
    frac = np.linspace(0.02, 0.97, 300)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # ---- left: production -> steady-state H2O2, both scenarios ----
    a = ax[0]
    for Km, col, lab in [(KM_A, NAVY, "Scenario A  ($K_m$ = 28 µM)"),
                         (KM_B, TAN,  "Scenario B  ($K_m$ = 262 µM)")]:
        H = np.array([H_ss(f * VMAX, Km) for f in frac])
        a.plot(100 * frac, H, color=col, lw=2.4, label=lab)
        rs = 100 * r_star(Km)
        a.axvline(rs, color=col, ls=":", lw=1.2)
        a.annotate(f"{rs:.1f}%", xy=(rs, H_MID), xytext=(rs, 4.5),
                   ha="center", fontsize=8.5, color=col,
                   arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    a.axhline(H_MID, color=RED, ls="--", lw=1.5)
    a.text(97, H_MID * 1.25, "48 µM — sensor midpoint",
           ha="right", fontsize=8, color=RED)
    a.axhline(H_LESION, color=GREY, ls=":", lw=1.2)
    a.text(97, H_LESION * 1.25, "250 µM — assumed lesional",
           ha="right", fontsize=8, color=GREY)
    a.set_yscale("log"); a.set_ylim(2, 2e4); a.set_xlim(0, 100)
    a.set_xlabel("lesion production, % of clearance capacity")
    a.set_ylabel("steady-state H$_2$O$_2$ (µM)")
    a.set_title("Clearance scenario changes the shutdown ratio",
                loc="left", color=NAVY, fontsize=9.5)
    a.legend(loc="upper left", fontsize=8)

    # ---- right: the same axis sent through the FLARE promoter response ----
    a = ax[1]
    for Km, col, lab in [(KM_A, NAVY, "Scenario A"), (KM_B, TAN, "Scenario B")]:
        H = np.array([H_ss(f * VMAX, Km) for f in frac])
        Pt = np.array([steady_state(min(h, 5e4))["P_trxB"] for h in H])
        a.plot(100 * frac, Pt, color=col, lw=2.4, label=lab)
        a.axvline(100 * r_star(Km), color=col, ls=":", lw=1.2)
    a.axhline(steady_state(H_MID)["P_trxB"], color=RED, ls="--", lw=1.4)
    a.text(2, steady_state(H_MID)["P_trxB"] + 0.03,
           "P$_{trxB}$ at the 48 µM midpoint", fontsize=8, color=RED)
    a.set_xlim(0, 100); a.set_ylim(0, 0.95)
    a.set_xlabel("lesion production, % of clearance capacity")
    a.set_ylabel("P$_{trxB}$")
    a.set_title("Same 48 µM sensor, different lesion coupling", loc="left",
                color=NAVY, fontsize=9.5)
    a.legend(loc="lower right")

    fig.tight_layout(); save(fig, "fig01_scenarios"); plt.show()


# =============================================================================
# FIGURE 2  -  Section 4.1
# Cell density changes total clearance, and therefore the midpoint boundary
# =============================================================================
def fig02():
    """
    Section 4.1, Figure 2  (two panels, as in the document).
    "Cell density changes total clearance through V_max = q_cell N.  It
     therefore changes where the 48 uM midpoint is reached.  After calibrating
     both scenarios to 250 uM at 6e7 CFU/mL, the midpoint boundary is about
     8.5e7 CFU/mL in Scenario A and 18.9e7 CFU/mL in nominal Scenario B."
    """
    print("Figure 2 ...")
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- left: dose sets clearance capacity ----------------------------------
    a = ax[0]
    N = np.logspace(7, 9, 300)
    a.loglog(N, VMAX * N / N_DESIGN, color=NAVY, lw=2.2, label="$V_{max}(N)$")
    P0A, P0B = P_les_for(H_LESION, KM_A), P_les_for(H_LESION, KM_B)
    a.axhline(P0A, color=MID, ls="--", lw=1.8,
              label="$P_{les}$ calibrated for Scenario A")
    a.axhline(P0B, color=MID, ls=":", lw=1.8,
              label="$P_{les}$ calibrated for Scenario B")
    a.axvline(N_DESIGN, color=INK, ls="-.", lw=1.3, label="design dose")
    a.set_xlabel("applied cell density (CFU/mL)")
    a.set_ylabel("rate (µM h$^{-1}$)")
    a.set_title("Dose sets clearance capacity", loc="left",
                color=NAVY, fontsize=9.5, pad=15)
    a.text(0.5, 1.012, "250 µM lesion calibration assumed", transform=a.transAxes,
           ha="center", fontsize=7, color=GREY)
    a.legend(loc="upper left", fontsize=7.5)

    # ---- right: compartment choice changes dose tolerance --------------------
    a = ax[1]
    Nr = np.logspace(np.log10(3e7), 9, 70)
    for Km, col, ls, lab in [(KM_A, NAVY, "-",  "Scenario A"),
                             (KM_B, TAN,  "--", "Scenario B")]:
        P0 = P_les_for(H_LESION, Km)
        # where production outruns clearance capacity H is unbounded, so the
        # sensor simply sits at its saturating output
        y = [steady_state(min(H_ss(P0, Km, VMAX * n / N_DESIGN), 5e3))["P_trxB"]
             for n in Nr]
        a.semilogx(Nr, y, color=col, ls=ls, lw=2.4, label=lab)
    for nb, col, ls, lab in [(8.5e7, NAVY, ":",  "A threshold: 8.5×10$^7$"),
                             (18.9e7, TAN, "-.", "B threshold: 18.9×10$^7$")]:
        a.axvline(nb, color=col, ls=ls, lw=1.4, label=lab)
    a.set_xlabel("applied cell density (CFU/mL)")
    a.set_ylabel("P$_{trxB}$")
    a.set_ylim(0, 0.9)
    a.set_title("Compartment choice changes dose tolerance", loc="left",
                color=NAVY, fontsize=9.5, pad=15)
    a.text(0.5, 1.012, "250 µM lesion calibration assumed", transform=a.transAxes,
           ha="center", fontsize=7, color=GREY)
    a.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(); save(fig, "fig02_cell_density"); plt.show()
    for Km, nm in ((KM_A, "A"), (KM_B, "B")):
        P0 = P_les_for(H_LESION, Km)
        nb = N_DESIGN * P0 * (Km + H_MID) / H_MID / VMAX
        print(f"    Scenario {nm}: P_les = {P0:.1f} µM/h, "
              f"midpoint boundary = {nb/1e7:.1f}e7 CFU/mL")
    print("    document: 8.5e7 (A) and 18.9e7 CFU/mL (B)")


# =============================================================================
# FIGURE 3  -  Section 4.3
# Gate 1 depends on the product phi_Fe x k_Fe  (k_PerR,eff)
# =============================================================================
def fig03(phis=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0)):
    """
    Section 4.3, Figure 3  (two panels, as in the document).
    "Gate 1 depends on the product phi_Fe x k_Fe.  The plot shows how changing
     the Fe-loaded fraction would move the response when k_Fe is fixed at
     12.5 uM^-1 h^-1.  In the revised uncertainty analysis, phi_Fe is not
     varied on its own.  We vary the identifiable combined parameter
     k_PerR,eff instead."
    """
    print("Figure 3 ...")
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.5))
    shades = ["#cfe8f5", "#9FD7ED", MID, "#3d7fb8", "#1f5e93", NAVY]
    xs = np.logspace(-2, np.log10(1500), 46)
    mids = []

    a = ax[0]
    for col, phi in zip(shades, phis):
        p = dict(P); p["phi_Fe"] = phi
        ys = np.array([steady_state(x, p)["P_trxB"] for x in xs])
        K, _ = fit_hill(xs, ys)
        mids.append(K)
        nominal = abs(phi - P["phi_Fe"]) < 1e-12
        a.semilogx(xs, ys, color=TAN if nominal else col,
                   lw=3.0 if nominal else 1.8, label=f"{phi:g}")
    a.set_ylim(0, 1.0)
    a.set_xlabel("applied H$_2$O$_2$ (µM)"); a.set_ylabel("P$_{trxB}$")
    a.set_title("Dose–response vs Fe-loaded fraction $\\varphi_{Fe}$",
                loc="left", color=NAVY, fontsize=9.5)
    a.legend(title="$\\varphi_{Fe}$", loc="upper left", ncol=2, fontsize=8,
             title_fontsize=8, frameon=False)

    a = ax[1]
    a.axhspan(40, 60, color=TAN, alpha=0.18, lw=0)
    a.semilogx(phis, mids, color=NAVY, lw=2.2, marker="o", ms=5)
    i = list(phis).index(P["phi_Fe"])
    a.scatter([phis[i]], [mids[i]], s=110, color=TAN, edgecolor=NAVY,
              lw=1.4, zorder=6)
    a.annotate("nominal\n$\\varphi_{Fe}$ = 0.05", xy=(phis[i], mids[i]),
               xytext=(0.13, mids[i] + 42), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.0))
    a.set_xlabel("$\\varphi_{Fe}$"); a.set_ylabel("output midpoint (µM)")
    a.set_title("Where the switch sits", loc="left", color=NAVY, fontsize=9.5)

    fig.tight_layout(); save(fig, "fig03_gate1"); plt.show()
    print("    phi_Fe :  " + "  ".join(f"{p:g}" for p in phis))
    print("    midpoint: " + "  ".join(f"{m:.0f}" for m in mids) + " uM")


# =============================================================================
# FIGURE 4  -  Section 4.5
# The two gates separately, then combined
# =============================================================================
def fig04():
    print("Figure 4 ...")
    xs = np.logspace(-2, 3.2, 150)
    res = [steady_state(x) for x in xs]
    spx_tot = np.array([r["Spx_tot"] for r in res])
    f_ox    = np.array([r["f_ox"] for r in res])
    spx_ox  = np.array([r["Spx_ox"] for r in res])
    Pt      = np.array([r["P_trxB"] for r in res])
    K, n = fit_hill(xs, Pt)

    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.0))

    a = ax[0, 0]
    a.semilogx(xs, spx_tot, color=NAVY, lw=2.4)
    a.fill_between(xs, 0, spx_tot, color=ICE, alpha=0.45)
    a.set_ylabel("Spx total (µM)")
    a.set_title("Gate 1 — Spx abundance", loc="left", color=MID, fontsize=9.5)

    a = ax[0, 1]
    a.semilogx(xs, f_ox, color=TAN, lw=2.4)
    a.fill_between(xs, 0, f_ox, color=TAN, alpha=0.20)
    a.set_ylabel("Spx$_{ox}$ / Spx total"); a.set_ylim(0, 1)
    a.set_title("Gate 2 — oxidised fraction", loc="left", color=TAN, fontsize=9.5)

    a = ax[1, 0]
    a.semilogx(xs, spx_ox, color=NAVY, lw=2.4)
    a.fill_between(xs, 0, spx_ox, color=ICE, alpha=0.45)
    a.set_xlabel("applied H$_2$O$_2$ (µM)"); a.set_ylabel("Spx$_{ox}$ (µM)")
    a.set_title("Gate 1 × Gate 2 — active Spx", loc="left", color=NAVY, fontsize=9.5)

    a = ax[1, 1]
    a.semilogx(xs, Pt, color=NAVY, lw=2.6)
    a.axvline(K, color=TAN, ls="--", lw=1.5)
    a.text(K * 1.25, 0.20, f"midpoint {K:.1f} µM\n$n_{{eff}}$ = {n:.2f}",
           fontsize=8.5, color=TAN)
    a.set_xlabel("applied H$_2$O$_2$ (µM)"); a.set_ylabel("P$_{trxB}$")
    a.set_ylim(0, 0.95)
    a.set_title("Output — P$_{trxB}$ activity", loc="left", color=NAVY, fontsize=9.5)

    fig.tight_layout(); save(fig, "fig04_two_gates"); plt.show()
    print(f"    fitted midpoint {K:.2f} uM, n_eff {n:.3f}")


# =============================================================================
# FIGURE 5  -  Section 4.5
# k_ox controls how easily the oxidant field flips the CXXC switch
# =============================================================================
def fig05(koxs=(0.02, 0.06, 0.15, 0.4, 1.0)):
    """
    Section 4.5, Figure 5  (two panels, as in the document).
    "k_ox controls how easily the oxidant signal activates the Spx CXXC switch.
     Across the plotted range, the fitted midpoint moves from about 76 uM to
     24 uM.  k_ox has not been measured directly, so it remains a high-priority
     parameter to test."
    """
    print("Figure 5 ...")
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.5))
    shades = ["#EBD0B0", TAN, "#a8723a", MID, NAVY]
    xs = np.logspace(-2, np.log10(1500), 46)
    mids = []

    a = ax[0]
    for col, k in zip(shades, koxs):
        p = dict(P); p["k_ox"] = k
        ys = np.array([steady_state(x, p)["P_trxB"] for x in xs])
        K, _ = fit_hill(xs, ys)
        mids.append(K)
        nominal = abs(k - P["k_ox"]) < 1e-12
        a.semilogx(xs, ys, color=col, lw=3.0 if nominal else 1.8, label=f"{k:g}")
    a.set_ylim(0, 1.0)
    a.set_xlabel("applied H$_2$O$_2$ (µM)"); a.set_ylabel("P$_{trxB}$")
    a.set_title("Dose–response vs redox-gate strength $k_{ox}$",
                loc="left", color=TAN, fontsize=9.5)
    a.legend(title="$k_{ox}$", loc="upper left", fontsize=8,
             title_fontsize=8, frameon=False)

    a = ax[1]
    a.semilogx(koxs, mids, color=TAN, lw=2.2, marker="o", ms=6)
    a.set_xlabel("$k_{ox}$  ((eq-µM)$^{-1}$h$^{-1}$)")
    a.set_ylabel("output midpoint (µM)")
    a.set_title(f"{mids[0]/mids[-1]:.1f}-fold spread — circuit-level "
                f"calibration priority", loc="left", color=TAN, fontsize=9.5)

    fig.tight_layout(); save(fig, "fig05_kox"); plt.show()
    print(f"    midpoint range {mids[0]:.1f} -> {mids[-1]:.1f} uM "
          f"({mids[0]/mids[-1]:.1f}-fold); document: about 76 -> 24")


# =============================================================================
# FIGURE 6  -  Section 4.6
# Left: onset and withdrawal dynamics.  Right: decay once the source stops.
# =============================================================================
def fig06():
    """
    Section 4.6, Figure 6  (two panels, as in the document).
    "Left: the promoter reaches half-rise in 0.67 h after a step to 100 uM H2O2
     and half-decays in 0.19 h after sudden withdrawal.  Right: after the H2O2
     source stops, the level falls from 250 to 48 uM in 4.1 h in Scenario A and
     10.5 h in Scenario B at the full cell dose.  At 1% residual cell density,
     the same crossing takes 17.2 and 43.8 days."
    """
    print("Figure 6 ...")
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))

    # ---- left: intrinsic onset after a step to 100 uM -----------------------
    a = ax[0]
    # pre-equilibrate at the OFF input for 2 h, then step to 100 uM at t = 0,
    # exactly as verify() does -- starting the states at zero would slow the
    # rise and report a t_half above the document's 0.67 h
    T, Y0, P0t = simulate(lambda tt: 100.0 if tt >= 2.0 else 0.01,
                          t_end=14.0, n=2800, breakpoints=[2.0])
    m = T >= 2.0
    t = T[m] - 2.0
    Y = [row[m] for row in Y0]
    Pt = P0t[m]
    PerR, S_red, S_ox = Y[0], Y[1], Y[2]
    S_tot = S_red + S_ox
    for series, col, lab in [
            ((P["PerR_tot"] - PerR) / P["PerR_tot"], MID,  "PerR inactivated"),
            (S_tot / S_tot[-1],                      TAN,  "Spx total (norm.)"),
            (S_ox / S_ox[-1],                        ICE,  "Spx$_{ox}$ (norm.)"),
            ((Pt - Pt[0]) / (Pt[-1] - Pt[0]),        NAVY, "P$_{trxB}$ (norm.)")]:
        a.plot(t, series, color=col, lw=3.0 if lab.startswith("P$_{trxB}") else 2.0,
               label=lab)
    t_half = t[np.argmax((Pt - Pt[0]) / (Pt[-1] - Pt[0]) >= 0.5)]
    a.axvline(t_half, color=MID, ls=":", lw=1.4)
    a.set_xlim(0, 12); a.set_ylim(0, 1.05)
    a.set_xlabel("hours after step to 100 µM"); a.set_ylabel("normalised value")
    a.set_title(f"Intrinsic onset: t$_{{1/2}}$ $\\approx$ {t_half:.2f} h",
                loc="left", color=NAVY, fontsize=9.5)
    a.legend(loc="lower right", fontsize=8)

    # ---- right: clearance through the 48 uM midpoint ------------------------
    a = ax[1]
    hrs = np.logspace(-2, np.log10(3000), 900)
    rows = []
    for Km, nm, ls in ((KM_A, "A", "-"), (KM_B, "B", "--")):
        for frac, col, dens in ((1.0, NAVY if nm == "A" else TAN, "full dose"),
                                (0.01, MID if nm == "A" else RED, "1% dose")):
            Vm = VMAX * frac

            def f(tt, y):
                H = max(y[0], 0.0)
                return [-Vm * H / (Km + H)]

            def hit(tt, y):
                return y[0] - H_MID
            hit.terminal, hit.direction = False, -1   # record, do not stop
            s = solve_ivp(f, (0, hrs[-1]), [H_LESION], t_eval=hrs,
                          method="LSODA", rtol=1e-10, atol=1e-12, events=hit)
            H = np.maximum(s.y[0], 1e-3)
            a.loglog(s.t, H, color=col, ls=ls, lw=2.0,
                     label=f"Scenario {nm}, {dens}")
            tc = s.t_events[0][0] if len(s.t_events[0]) else np.nan
            rows.append((nm, dens, tc))
    a.axhline(H_MID, color=MID, ls=":", lw=1.4)
    a.set_ylim(1, 400); a.set_xlim(1e-2, 3e3)
    a.set_xlabel("hours after source stops"); a.set_ylabel("H$_2$O$_2$ (µM)")
    a.set_title("Clearance through the 48 µM sensor midpoint",
                loc="left", color=NAVY, fontsize=9.5)
    a.legend(loc="lower left", fontsize=7.5, framealpha=1.0,
             facecolor="white", edgecolor=GRID)

    fig.tight_layout(); save(fig, "fig06_dynamics"); plt.show()
    t2, _, P2 = simulate(lambda tt: 100.0 if tt < 12 else 0.0, t_end=24.0, n=4000)
    lo, hi = P2[0], P2[np.argmax(t2 >= 11.9)]
    after = t2 >= 12.0
    hd = t2[after][np.argmax(P2[after] <= lo + 0.5 * (hi - lo))] - 12.0
    print(f"    half-rise {t_half:.2f} h, half-decay {hd:.2f} h "
          f"(document: 0.67 h and 0.19 h)")
    for nm, dens, tc in rows:
        pretty = f"{tc:.1f} h" if tc < 48 else f"{tc/24:.1f} d"
        print(f"    Scenario {nm}, {dens:<9s}: 250 -> 48 µM in {pretty}")
    print("    document: 4.1 h / 10.5 h at full dose, 17.2 d / 43.8 d at 1%")


# =============================================================================
# FIGURE 7  -  Section 4.7
# Lesion-circuit simulation for different clearance capacities
# =============================================================================
# Lesion-resolution time constant.  Section 4.7 does not print the resolution
# curve; it only states that the 1x Scenario-A case reaches the 48 uM midpoint
# "at day 5 by construction".  TAU_RES = 335 h (~14 days) is the single-exponential
# constant that reproduces that anchor AND the whole section 4.7 day-count table
# for both scenarios and every clearance multiple (see days_above_midpoint).
TAU_RES = 335.0


def coupled_lesion(Km, mult, t_end=24 * 22, n=5000, tau_res=TAU_RES):
    """
    Section 4.7.  H2O2 and the circuit are integrated TOGETHER:

        dH/dt = P_les(t) - Vmax_eff * H/(Km + H),
        P_les(t) = P_les(0) * exp(-t / tau_res)

    plus the four FLARE states driven by H(t).  The coupling matters.  The
    circuit needs ~0.67 h to respond, so a fast-clearing strain still shows a
    transient PEAK in P_trxB before the signal is removed -- the document's
    "peaks near 0.52" (Scenario A) and "near 0.66" (Scenario B) at 5x
    clearance.  A quasi-steady-state treatment misses those peaks entirely.
    """
    P0 = P_les_for(H_LESION, Km)
    Vm = VMAX * mult
    p = P

    def f(t, y):
        H, PerR, S_red, S_ox, Trx = y
        P_les = P0 * np.exp(-t / tau_res)
        dH = P_les - Vm * max(H, 0.0) / (Km + max(H, 0.0))
        d = rhs(t, [PerR, S_red, S_ox, Trx], lambda tt: max(H, 0.0), p)
        return [dH] + d

    y0 = [H_LESION, p["PerR_tot"], 0.0, 0.0, p["Trx_bas"]]
    s = solve_ivp(f, (0, t_end), y0, t_eval=np.linspace(0, t_end, n),
                  method="LSODA", rtol=1e-8, atol=1e-11)
    H = np.maximum(s.y[0], 0.0)
    P_trxB = p["P_basal"] + (1 - p["P_basal"]) * hill(s.y[3], p["K_out"],
                                                      p["n_out"])
    return s.t, H, P_trxB


def days_above_midpoint(Km, mult, tau_res=TAU_RES, t_end=24 * 60):
    """
    Section 4.7 day-count table ("Days A" / "Days B").

    This is a DIFFERENT initial condition from the Figure-7 trajectories.  The
    table asks how long a strain of a given clearance capacity sits above the
    48 uM midpoint while the lesion resolves, so each strain STARTS at its own
    steady state H_ss(mult) -- not at the applied 250 uM.  That is what the
    document means by "A value of 0 d in the table means the system does not
    spend positive time above the 48 uM midpoint.  The 1.42x case begins exactly
    at the boundary, while the 2x and 5x cases start below it."

    The Figure-7 panels, by contrast, start at the applied lesional 250 uM,
    which is what produces the transient P_trxB peaks and the "250 -> about
    6 uM within two hours" statement in section 4.8.
    """
    P0 = P_les_for(H_LESION, Km)
    Vm = VMAX * mult
    H0 = H_ss(P0, Km, Vm)
    if H0 <= H_MID:                     # starts at or below the boundary
        return 0.0

    def f(t, y):
        H = max(y[0], 0.0)
        return [P0 * np.exp(-t / tau_res) - Vm * H / (Km + H)]

    def hit(t, y):
        return y[0] - H_MID
    hit.terminal, hit.direction = True, -1

    s = solve_ivp(f, (0, t_end), [H0], method="LSODA",
                  rtol=1e-10, atol=1e-12, events=hit)
    return s.t_events[0][0] / 24 if len(s.t_events[0]) else np.nan


def fig07():
    """
    Section 4.7, Figure 7  (2 x 2: scenario x clearance, as in the document).
    "Lesion-circuit simulations under both K_m interpretations.  In Scenario A,
     ATCC 8014 reaches the 48 uM midpoint at day 5 by construction.  A
     fivefold-clearance strain peaks near P_trxB = 0.52 and lowers H2O2 to
     6.3 uM within 2 h.  In Scenario B, the 1x strain reaches the midpoint at
     about 16.3 days.  The fivefold-clearance case peaks near 0.66 and crosses
     48 uM after about 3.3 h."

    Each panel carries P_trxB on the left axis (solid) and H2O2 on a log right
    axis (dashed).  The 1x panels run in days; the 5x panels run in hours,
    because the fast-clearance transient is over within a few hours.
    """
    print("Figure 7 ...")
    panels = [(KM_A, 1.0, 10.0 * 24, "days",  "Scenario A: ATCC 8014 (1×)"),
              (KM_A, 5.0,        3.0, "hours", "Scenario A: 5× clearance"),
              (KM_B, 1.0, 20.0 * 24, "days",  "Scenario B: ATCC 8014 (1×)"),
              (KM_B, 5.0,        8.0, "hours", "Scenario B: 5× clearance")]

    fig, ax = plt.subplots(2, 2, figsize=(9.8, 6.2))
    out = []
    for k, (Km, mult, t_end, unit, title) in enumerate(panels):
        a = ax[k // 2, k % 2]
        t, H, Pt = coupled_lesion(Km, mult, t_end=t_end, n=4000)
        x = t / 24 if unit == "days" else t

        a.plot(x, Pt, color=NAVY, lw=2.2, label="P$_{trxB}$")
        a.axhline(0.50, color=MID, ls=":", lw=1.2)
        a.set_xlim(0, x[-1]); a.set_ylim(0, 0.88)
        a.set_ylabel("P$_{trxB}$"); a.set_xlabel(unit.capitalize())
        a.set_title(title, loc="center", color=NAVY, fontsize=10, pad=16)
        a.text(0.5, 1.012, "250 µM lesion calibration assumed",
               transform=a.transAxes, ha="center", fontsize=7, color=GREY)

        b = a.twinx()
        b.semilogy(x, H, color=NAVY, ls="--", lw=2.0, label="H$_2$O$_2$")
        b.set_ylabel("H$_2$O$_2$ (µM)")
        b.set_ylim(4, 330)          # one shared H2O2 scale across all panels
        b.grid(visible=False)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = b.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
        out.append((title, Pt.max(), t[int(np.argmax(Pt))], H[-1], Pt[-1]))

    fig.tight_layout(); save(fig, "fig07_chassis"); plt.show()

    print("    panel                         peak P_trxB   at      end H2O2   end P_trxB")
    for title, pk, tp, hend, pend in out:
        print(f"    {title:<28s}   {pk:.3f}   {tp:6.2f} h   "
              f"{hend:7.1f}    {pend:.3f}")
    print("    document: 5x peaks near 0.52 (A) and 0.66 (B)")

    # ---- section 4.7 table: H_ss and days above the 48 uM midpoint ----------
    print()
    print("    Section 4.7 table (H_ss from Eq 2; days above 48 uM measured on")
    print("    the lesion-resolution curve, each strain starting at its own H_ss)")
    print("    clearance            H_ss A   Days A     H_ss B   Days B")
    doc = {1.0: (250, 5.0, 250, 16.3), 1.1: (125, 3.7, 209, 14.9),
           1.2: (84, 2.4, 180, 13.7), 1.3: (63, 1.3, 158, 12.5),
           1.42: (48, 0.0, 137, 11.3), 2.0: (23, 0.0, 85, 6.4),
           5.0: (6, 0.0, 28, 0.0)}
    for m, (dHA, dDA, dHB, dDB) in doc.items():
        hA = H_ss(P_les_for(H_LESION, KM_A), KM_A, VMAX * m)
        hB = H_ss(P_les_for(H_LESION, KM_B), KM_B, VMAX * m)
        print(f"     {m:5.2f}x            {hA:6.1f}  "
              f"{days_above_midpoint(KM_A, m):5.1f} d     "
              f"{hB:6.1f}  {days_above_midpoint(KM_B, m):5.1f} d      "
              f"(document: {dHA} / {dDA} d , {dHB} / {dDB} d)")


# =============================================================================
# FIGURE 8  -  Section 6
# Local +/-10% sensitivity on two different endpoints
# =============================================================================
def fig08():
    """
    Section 6, Figure 8  (two panels, as in the document).
    "Left: P_trxB is tested at a fixed 100 uM H2O2.  Because H2O2 is clamped,
     the clearance equations are bypassed and m_prot, Npx_SA and K_m,Npx have
     no effect on this endpoint.  Right: the endpoint is the lesion-production
     rate needed to reach the 48 uM midpoint.  Here m_prot and Npx_SA change
     V_max directly, while K_m,Npx is especially important in Scenario B."
    """
    print("Figure 8 ...")
    xs = np.logspace(-2, np.log10(1500), 40)

    # ---- left endpoint: P_trxB at a clamped 100 uM --------------------------
    base = steady_state(100.0)["P_trxB"]
    left = ["k_syn", "K_out", "k_clp", "n_out", "min_clp", "n_clp", "f_in",
            "k_ox", "chi_OX", "delta_spx", "n_perR", "K_perR", "k_Fe", "phi_Fe",
            "K_clp", "PerR_tot", "delta_Per", "k_trx", "Trx_bas",
            "m_prot", "Km_Npx", "Npx_SA"]
    nice = {"chi_OX": "χ_OX", "delta_spx": "δ_spx", "delta_Per": "δ_Per",
            "phi_Fe": "φ_Fe", "k_syn": "k_syn_spx",
            "Trx_bas": "Trx_bas (Trx_max=1.8×)"}
    up, dn = [], []
    for k in left:
        for sgn, store in ((1.10, up), (0.90, dn)):
            if k in ("m_prot", "Km_Npx", "Npx_SA"):
                store.append(0.0)              # clearance only; H2O2 is clamped
                continue
            p = dict(P); p[k] = P[k] * sgn
            if k == "Trx_bas":
                p["Trx_max"] = 1.8 * p["Trx_bas"]
            store.append(100 * (steady_state(100.0, p)["P_trxB"] - base) / base)
    order = np.argsort([max(abs(u), abs(d)) for u, d in zip(up, dn)])

    # ---- right endpoint: lesion production needed to reach the midpoint -----
    # P_les* = V_max H*/(K_m + H*), where H* is the sensor midpoint.  Circuit
    # parameters enter through H*; m_prot and Npx_SA scale V_max directly and
    # K_m,Npx enters the Michaelis term.
    def p_les_star(Km, Vm=VMAX, p=None):
        if p is None:
            Hs = H_MID
        else:
            Hs, _ = fit_hill(xs, np.array([steady_state(x, p)["P_trxB"]
                                           for x in xs]))
        return Vm * Hs / (Km + Hs)

    right = ["m_prot", "Npx_SA", "Km_Npx", "n_clp", "k_clp", "n_out", "K_clp",
             "k_syn", "K_out", "k_ox", "chi_OX", "f_in", "K_perR"]
    baseA, baseB = p_les_star(KM_A), p_les_star(KM_B)
    effA, effB = [], []
    for k in right:
        dA, dB = [], []
        for sgn in (1.10, 0.90):
            if k in ("m_prot", "Npx_SA"):
                vA = p_les_star(KM_A, VMAX * sgn)
                vB = p_les_star(KM_B, VMAX * sgn)
            elif k == "Km_Npx":
                vA = p_les_star(KM_A * sgn)
                vB = p_les_star(KM_B * sgn)
            else:
                p = dict(P); p[k] = P[k] * sgn
                vA = p_les_star(KM_A, VMAX, p)
                vB = p_les_star(KM_B, VMAX, p)
            dA.append(abs(100 * (vA - baseA) / baseA))
            dB.append(abs(100 * (vB - baseB) / baseB))
        effA.append(max(dA)); effB.append(max(dB))
    ro = np.argsort([max(a, b) for a, b in zip(effA, effB)])

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.6))

    a = ax[0]
    y = np.arange(len(left))
    a.barh(y, [up[i] for i in order], color=NAVY, height=0.62, label="+10%")
    a.barh(y, [dn[i] for i in order], color=TAN,  height=0.62, label="−10%")
    a.set_yticks(y)
    a.set_yticklabels([nice.get(left[i], left[i]) for i in order], fontsize=8)
    a.axvline(0, color=MID, lw=1.0); a.grid(axis="y", visible=False)
    a.set_xlabel("change in P$_{trxB}$ at 100 µM (%)")
    a.set_title("Intrinsic fixed-input endpoint", loc="left",
                color=NAVY, fontsize=9.5)
    a.legend(loc="lower right", fontsize=8)

    a = ax[1]
    y2 = np.arange(len(right))
    a.barh(y2 + 0.19, [effA[i] for i in ro], height=0.36, color=NAVY,
           label="Scenario A")
    a.barh(y2 - 0.19, [effB[i] for i in ro], height=0.36, color=TAN,
           label="Scenario B")
    a.set_yticks(y2)
    a.set_yticklabels([nice.get(right[i], right[i]) for i in ro], fontsize=8.5)
    a.grid(axis="y", visible=False)
    a.set_xlabel("max |change| in shutdown $P_{les}$* (%)")
    a.set_title("Coupled sensor + clearance endpoint", loc="left",
                color=NAVY, fontsize=9.5)
    a.legend(loc="lower right", fontsize=8)

    fig.tight_layout(); save(fig, "fig08_local_sensitivity"); plt.show()
    print(f"    largest local effect on P_trxB: "
          f"{max(max(abs(u), abs(d)) for u, d in zip(up, dn)):.2f}%  "
          f"(document: no more than 3.33%)")
    print("    coupled endpoint: m_prot and Npx_SA give a full ±10%; "
          f"K_m,Npx gives {effA[right.index('Km_Npx')]:.1f}% (A) and "
          f"{effB[right.index('Km_Npx')]:.1f}% (B)")


# =============================================================================
# FIGURES 9 & 10  -  Section 6
# Global sensitivity: 16 quantities, Latin-hypercube, two ClpXP versions
# =============================================================================
N_LHS = 600          # <-- set to 4000 to reproduce the published statistics
SEED  = 20260909

# prior ranges, as tabulated in section 6
PRIORS = {
    "K_out":      (0.06,   6.0,   "log"),
    "k_ox":       (0.006,  0.60,  "log"),
    "k_clp":      (2.0,   18.0,   "log"),
    "K_clp":      (0.15,  15.0,   "log"),
    "k_PerR_eff": (0.0625, 6.25,  "log"),
    "PerR_tot":   (0.60,  12.0,   "log"),
    "Trx_bas":    (0.05,   5.0,   "log"),
    "delta_Per":  (0.15,   1.50,  "log"),
    "delta_spx":  (0.12,   1.05,  "log"),
    "P_basal":    (0.001,  0.05,  "log"),
    "f_in":       (0.03,   0.35,  "log"),
    "n_clp":      (1.0,    3.0,   "lin"),
    "n_out":      (1.0,    2.0,   "lin"),
    "K_perR":     (0.116, 11.6,   "log"),
    "n_perR":     (0.5,    2.0,   "lin"),
    "min_clp":    (0.0,    0.0969,"lin"),
}


def _sample(n=N_LHS, seed=SEED):
    keys = list(PRIORS)
    u = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    out = {}
    for j, k in enumerate(keys):
        lo, hi, mode = PRIORS[k]
        out[k] = (np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
                  if mode == "log" else lo + u[:, j] * (hi - lo))
    return keys, out


def run_gsa(n=N_LHS, ox_driven=False, seed=SEED, verbose=True):
    """
    Returns (keys, X, midpoints, n_effs, off_fracs).
    ox_driven=False -> ClpXP inhibited by intracellular H2O2 (Eq 6, nominal)
    ox_driven=True  -> ClpXP inhibited by the oxidant field OX instead
    """
    keys, S = _sample(n, seed)
    xs = np.logspace(-2, np.log10(1500), 40)
    mids, neffs, offs, X = [], [], [], []
    K_CLP_OX = 5.61      # reparameterised half-point for the OX-driven version

    for i in range(n):
        p = dict(P)
        p["K_out"]     = S["K_out"][i]
        p["k_ox"]      = S["k_ox"][i]
        p["k_clp"]     = S["k_clp"][i]
        p["K_clp"]     = S["K_clp"][i] if not ox_driven else S["K_clp"][i] * K_CLP_OX / 1.5
        p["PerR_tot"]  = S["PerR_tot"][i]
        p["Trx_bas"]   = S["Trx_bas"][i]
        p["Trx_max"]   = 1.8 * S["Trx_bas"][i]     # fixed 1.8 ratio
        p["delta_Per"] = S["delta_Per"][i]
        p["delta_spx"] = S["delta_spx"][i]
        p["P_basal"]   = S["P_basal"][i]
        p["f_in"]      = S["f_in"][i]
        p["n_clp"]     = S["n_clp"][i]
        p["n_out"]     = S["n_out"][i]
        p["K_perR"]    = S["K_perR"][i]
        p["n_perR"]    = S["n_perR"][i]
        p["min_clp"]   = S["min_clp"][i]
        # Gate 1 enters only as the product; hold k_Fe, vary phi_Fe to match
        p["phi_Fe"]    = S["k_PerR_eff"][i] / p["k_Fe"]
        p["_ox_clp"]   = ox_driven

        try:
            ys = np.array([_ss_gsa(x, p) for x in xs])
            if ys.max() - ys.min() < 0.02:
                continue
            K, ne = fit_hill(xs, ys)
            if not (1e-2 < K < 1e5) or not (0.2 < ne < 12):
                continue
            mids.append(K); neffs.append(ne)
            # "clean OFF state" (section 6): the resting output stays below 5%
            # of that sample's own saturating output.  The document quotes the
            # fraction (92.8%) but not the criterion, so it is stated here.
            offs.append(ys[0] / max(ys[-1], 1e-12))
            X.append([S[k][i] for k in keys])
        except Exception:
            continue
        if verbose and (i + 1) % max(n // 10, 1) == 0:
            print(f"    {i+1}/{n} samples ...")
    return (keys, np.array(X), np.array(mids), np.array(neffs),
            np.array(offs))


def _rhs_gsa(t, y, I_of_t, p):
    """As rhs(), but ClpXP may be driven by OX instead of intracellular H2O2."""
    PerR, S_red, S_ox, Trx = y
    I = I_of_t(t)
    I_int = p["f_in"] * I
    OX    = p["chi_OX"] * I
    P_spx = 1.0 / (1.0 + (max(PerR, 0.0) / p["K_perR"])**p["n_perR"])
    driver = OX if p.get("_ox_clp") else I_int
    g_clp = 1.0 - (1.0 - p["min_clp"]) * hill(driver, p["K_clp"], p["n_clp"])
    d_tot = p["k_clp"] * g_clp + p["delta_spx"]
    v_ox = p["k_ox"] * OX * S_red
    v_re = p["k_trx"] * Trx * S_ox
    dPerR  = p["delta_Per"] * (p["PerR_tot"] - PerR) \
             - p["phi_Fe"] * p["k_Fe"] * I_int * PerR
    dS_red = p["k_syn"] * P_spx - d_tot * S_red - v_ox + v_re
    dS_ox  = -d_tot * S_ox + v_ox - v_re
    P_trxB = p["P_basal"] + (1 - p["P_basal"]) * hill(S_ox, p["K_out"], p["n_out"])
    dTrx   = p["k_rel"] * (p["Trx_bas"] + (p["Trx_max"] - p["Trx_bas"]) * P_trxB - Trx)
    return [dPerR, dS_red, dS_ox, dTrx]


def _ss_gsa(I, p, t_end=250.0):
    y0 = [p["PerR_tot"], 0.0, 0.0, p["Trx_bas"]]
    s = solve_ivp(_rhs_gsa, (0, t_end), y0, args=(lambda t: I, p),
                  method="LSODA", rtol=1e-7, atol=1e-10)
    S_ox = s.y[2, -1]
    return p["P_basal"] + (1 - p["P_basal"]) * hill(S_ox, p["K_out"], p["n_out"])


def fig09_10(n=N_LHS):
    print(f"Figures 9-10  ({n} LHS samples per ClpXP version) ...")
    print("  H2O2-driven ClpXP:")
    keys, XH, midH, neH, offH = run_gsa(n, ox_driven=False)
    print("  OX-driven ClpXP:")
    _,    XO, midO, neO, offO = run_gsa(n, ox_driven=True)

    # ---------------- Figure 9 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    a = ax[0]
    bins = np.linspace(-1.0, 3.3, 44)
    a.hist(np.log10(midH), bins=bins, color=MID, alpha=0.65,
           label="H$_2$O$_2$-driven ClpXP")
    a.hist(np.log10(midO), bins=bins, color=TAN, alpha=0.65,
           label="OX-driven ClpXP")
    a.axvline(np.log10(48.01), color=MID, ls=":", lw=1.6)
    a.set_xlabel("log10 fitted midpoint (µM applied)")
    a.set_ylabel("accepted samples")
    a.set_title("Revised midpoint distribution", loc="left",
                color=NAVY, fontsize=9.5)
    a.legend(loc="upper left", fontsize=8)

    a = ax[1]
    bins2 = np.linspace(0.8, 6.0, 46)
    a.hist(neH, bins=bins2, color=MID, alpha=0.65, label="H$_2$O$_2$-driven ClpXP")
    a.hist(neO, bins=bins2, color=TAN, alpha=0.65, label="OX-driven ClpXP")
    a.axvline(2.12, color=MID, ls=":", lw=1.6)
    a.set_xlim(0.8, 6)
    a.set_xlabel("effective Hill coefficient"); a.set_ylabel("accepted samples")
    a.set_title("Switch sharpness remains better constrained", loc="left",
                color=NAVY, fontsize=9.5)
    a.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); save(fig, "fig09_gsa_distributions"); plt.show()

    # ---------------- Figure 10 ----------------
    rhoH = [spearmanr(XH[:, j], np.log10(midH)).statistic for j in range(len(keys))]
    rhoO = [spearmanr(XO[:, j], np.log10(midO)).statistic for j in range(len(keys))]
    order = np.argsort([abs(r) for r in rhoH])[::-1]      # strongest at the top
    label = {"k_PerR_eff": "k_PerR,eff"}

    fig, a = plt.subplots(figsize=(8.0, 4.8))
    y = np.arange(len(keys))[::-1]
    a.barh(y + 0.19, [rhoH[i] for i in order], height=0.36, color=NAVY,
           label="H$_2$O$_2$-driven ClpXP")
    a.barh(y - 0.19, [rhoO[i] for i in order], height=0.36, color=TAN,
           label="OX-driven ClpXP")
    a.set_yticks(y)
    a.set_yticklabels([label.get(keys[i], keys[i]) for i in order], fontsize=8.5)
    a.axvline(0, color=MID, lw=0.9); a.grid(axis="y", visible=False)
    a.set_xlabel("Spearman rank correlation with log10(midpoint)")
    a.set_title(f"Revised global sensitivity ranking "
                f"({n} Latin-hypercube samples)", loc="center", color=NAVY)
    a.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); save(fig, "fig10_gsa_correlations"); plt.show()

    # the three correlations the Figure 10 caption quotes by name
    print("    rank correlation with log(midpoint)     H2O2-driven   OX-driven")
    for k in ("K_out", "k_ox", "K_perR", "n_perR"):
        j = keys.index(k) if isinstance(keys, list) else list(keys).index(k)
        print(f"      {k:<34s} {rhoH[j]:+8.2f}    {rhoO[j]:+8.2f}")
    print("      document: K_out and k_ox strongest; K_perR -0.17 / -0.19;")
    print("                n_perR |rho| < 0.01")

    print()
    print("  Claim                                        H2O2-driven   OX-driven")
    for lab, fH, fO in [
        ("midpoint in 10-500 uM",
         np.mean((midH > 10) & (midH < 500)), np.mean((midO > 10) & (midO < 500))),
        ("n_eff > 1.5",             np.mean(neH > 1.5),  np.mean(neO > 1.5)),
        ("clean OFF state (rest < 5% of saturating)",
         np.mean(offH < 0.05),   np.mean(offO < 0.05)),
        ("midpoint in 40-60 uM",
         np.mean((midH > 40) & (midH < 60)), np.mean((midO > 40) & (midO < 60))),
    ]:
        print(f"    {lab:42s} {100*fH:8.1f}%   {100*fO:8.1f}%")
    print("    document (4000 samples, H2O2-driven): 86.9% / 90.6% / 92.8% / 10.8%")
    print(f"    midpoint  median {np.median(midH):.1f} uM (OX {np.median(midO):.1f}), "
          f"5-95% {np.percentile(midH,5):.1f}-{np.percentile(midH,95):.0f} uM")
    print(f"    n_eff     median {np.median(neH):.2f}, "
          f"5-95% {np.percentile(neH,5):.2f}-{np.percentile(neH,95):.2f}")
    print("    document: midpoint median 74.8 uM (OX 73.5), 5-95% 7.7-509 uM;")
    print("              n_eff median 2.31, 5-95% 1.38-4.29")
    return keys, XH, midH, neH


# =============================================================================
# MAIN
# =============================================================================
def main(include_gsa=True):
    verify()
    print()
    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08()
    if include_gsa:
        fig09_10()
    print("\nAll figures written to", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main(include_gsa=True)
