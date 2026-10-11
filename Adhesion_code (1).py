#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
WLC  -  Worm-Like Chain force-extension curves for SasG and Aap
Reproduces the two force-extension figures (Figures 1-2).

Model: Marko-Siggia worm-like chain,  kBT = 4.1 pN.nm,  Lp = 0.4 nm
Domains: E (19.0 nm) x4, G5 (29.6 nm) x4, A domain (SasG 139 nm / Aap 211 nm)

------------------------------------------------------------------------------
HOW TO USE IN GOOGLE COLAB
------------------------------------------------------------------------------
1. Upload this file, or paste each SECTION into its own cell.
2. Run SECTION 0 first (imports and the save helper).  Both figures depend
   on it.
3. Each figure lives in its own section and its own function, fig01() and
   fig02().  The two sections do not share any variables, so either one can
   be run on its own after SECTION 0.  Run main() at the bottom for both.
4. Figures are saved as PNG at 300 dpi into ./wlc_figures/ and also shown
   inline.

Only numpy and matplotlib are required - both present in Colab by default.
No other files are needed; this script is self-contained.

------------------------------------------------------------------------------
WHAT SHOULD REPRODUCE
------------------------------------------------------------------------------
  Figure 1 (rupture force calculated by the WLC equation)
    E  domain   90.2% of contour length    273.5 pN
    G5 domain   93.1% of contour length    545.2 pN
    SasG A      93.8% of contour length    673.7 pN
    Aap A       94.8% of contour length    954.8 pN
    maximum extension   SasG 312.7 nm (of 333.4 nm),  Aap 384.3 nm (of 405.4 nm)
  Figure 2 (rupture force set as a threshold)
    thresholds E 275 pN,  G 550 pN,  A 950 pN
    plotted peaks 276.0 / 552.5 / 953.9 pN (first grid point at or above each)
Run verify() to check these.

------------------------------------------------------------------------------
FIGURE INDEX
------------------------------------------------------------------------------
  fig01()   SasG vs Aap sawtooth curves; each domain ruptures at a fixed
            fraction of the total contour length, and the trace stops at the
            final peak (maximum extension)
  fig02()   Aap only; domain-by-domain sawtooth with labelled E, G and A peaks
==============================================================================
"""

# =============================================================================
# SECTION 0  -  IMPORTS AND SAVE HELPER
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTDIR = "wlc_figures"
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTDIR, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("  saved ->", path)


# =============================================================================
# FIGURE 1  -  SasG vs Aap
# Rupture force is calculated by the WLC equation from the fraction of the
# total contour length reached; the trace stops at the final peak
# =============================================================================
kT = 4.1    # pN.nm
P = 0.4     # nm


def wlc(x, Lc):
    """Marko-Siggia WLC force (pN) at extension x for contour length Lc."""
    ratio = x / Lc
    if ratio >= 1.0:
        raise ValueError(f"Extension {x:.1f} nm reached the contour length {Lc:.1f} nm")
    return (kT / P) * (1 / (4 * (1 - ratio) ** 2) - 0.25 + ratio)


# length = contour length added when the domain unfolds (nm)
# frac   = fraction of the total contour length reached at rupture
e_dom = {'type': 'E', 'length': 19.0, 'frac': 0.902}
g5_dom = {'type': 'G5', 'length': 29.6, 'frac': 0.931}
a_sasg = {'type': 'A (SasG)', 'length': 139.0, 'frac': 0.938}
a_aap = {'type': 'A (Aap)', 'length': 211.0, 'frac': 0.948}

sasg_doms = [e_dom] * 4 + [g5_dom] * 4 + [a_sasg]
aap_doms = [e_dom] * 4 + [g5_dom] * 4 + [a_aap]


def pull(doms, name="", dx=0.1):
    """Stretch the chain domain by domain; returns extension, force, peaks."""
    xx = []
    ff = []
    tops = []
    pos = 0.0
    Ltot = 0.0

    print(f"Simulating {name}...")
    for n, dom in enumerate(doms):
        Ltot += dom['length']
        xb = Ltot * dom['frac']

        xpart = np.arange(pos, xb, dx)
        xpart = np.append(xpart, xb)
        fpart = np.array([wlc(xi, Ltot) for xi in xpart])

        xx.extend(xpart)
        ff.extend(fpart)
        tops.append((xb, fpart[-1], dom['type']))
        print(f"  Domain {n+1}: {dom['type']:<9} L_total = {Ltot:6.1f} nm  "
              f"rupture at x = {xb:6.1f} nm  F = {fpart[-1]:6.1f} pN")
        pos = xb

    print(f"  Maximum extension = {pos:.1f} nm ({pos / Ltot:.1%} of {Ltot:.1f} nm)\n")

    return np.array(xx), np.array(ff), tops


def fig01():
    print("Figure 1 ...")
    x1, f1, tops1 = pull(sasg_doms, "SasG")
    x2, f2, tops2 = pull(aap_doms, "Aap")

    # experimental-style noise, fixed seed so the figure is reproducible
    gen = np.random.default_rng(42)
    noise = 15
    f1_n = f1 + gen.normal(0, noise, len(f1))
    f2_n = f2 + gen.normal(0, noise, len(f2))

    fig = plt.figure(figsize=(12, 7))

    plt.plot(x1, f1_n, label='SasG', color='blue', alpha=0.6)
    plt.plot(x2, f2_n, label='Aap', color='red', alpha=0.6)

    plt.plot(x1, f1, color='black', linewidth=1, alpha=0.4)
    plt.plot(x2, f2, color='black', linewidth=1, alpha=0.4)

    for tops, col in [(tops1, 'blue'), (tops2, 'red')]:
        px, pf, _ = zip(*tops)
        plt.scatter(px, pf, color=col, edgecolor='black', zorder=5, s=30)

    plt.title("WLC force-extension curves of SasG and Aap\n"
              "Each domain ruptures at a fixed fraction of the total contour length", fontsize=14)
    plt.xlabel("Extension x (nm)", fontsize=12)
    plt.ylabel("Force (pN)", fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.ylim(-50, 1100)
    plt.tight_layout()
    save(fig, "fig01_sasg_vs_aap")
    plt.show()
    return tops1, tops2


# =============================================================================
# FIGURE 2  -  Aap only
# Domain-by-domain sawtooth; each domain ruptures when the force reaches its
# set threshold
# =============================================================================
kbt = 4.1   # pN.nm
lp = 0.4    # nm

le = 19.0   # nm, E domain
lg = 29.6   # nm, G5 domain
la = 211.0  # nm, Aap A domain

fe = 275.0  # pN, E rupture
fg = 550.0  # pN, G rupture
fa = 950.0  # pN, A rupture

reps = 4


def wlc2(xs, L, lp, kbt):
    """Marko-Siggia WLC force (pN); r is clipped to avoid infinity at L."""
    r = xs / L
    r = np.clip(r, 0, 0.9995)
    return (kbt / lp) * (1.0 / (4.0 * (1.0 - r)**2) - 0.25 + r)


def sawtooth(le, lg, la, fe, fg, fa):
    """Order: E1-E4 -> G1-G4 -> A.  Returns extension, force, peaks."""
    parts = []
    for k in range(1, reps + 1):
        parts.append({'L': le, 'rup': fe, 'label': f'E{k}', 'type': 'E'})
    for k in range(1, reps + 1):
        parts.append({'L': lg, 'rup': fg, 'label': f'G{k}', 'type': 'G'})
    parts.append({'L': la, 'rup': fa, 'label': 'A', 'type': 'A'})

    xall = []
    fall = []
    pk = []
    shift = 0.0

    for p in parts:
        Ld = p['L']
        fmax = p['rup']

        xs = np.linspace(0, Ld, 2000)
        fs = wlc2(xs, Ld, lp, kbt)

        hits = np.where(fs >= fmax)[0]

        if len(hits) > 0:
            j = hits[0]
            xcut = xs[:j + 1]
            fcut = fs[:j + 1]
            xb = xs[j]
            fb = fs[j]
        else:
            xcut = xs
            fcut = fs
            xb = xs[-1]
            fb = fs[-1]

        xall.extend(xcut + shift)
        fall.extend(fcut)

        pk.append({'x': xb + shift, 'f': fb, 'label': p['label'], 'type': p['type']})

        xall.extend([xb + shift, xb + shift])
        fall.extend([fb, 0.0])

        shift += Ld

    return np.array(xall), np.array(fall), pk


def fig02():
    print("Figure 2 ...")
    xa, fa_curve, pk_aap = sawtooth(le, lg, la, fe, fg, fa)

    fig, ax = plt.subplots(figsize=(14, 6))
    red = '#d62728'
    cols = {'E': '#2ca02c', 'G': '#ff7f0e', 'A': '#9467bd'}

    ax.plot(xa, fa_curve, color=red, lw=2)
    for p in pk_aap:
        c = cols[p['type']]
        ax.plot(p['x'], p['f'], 'o', color=c, markersize=6)
        ax.annotate(p['label'], xy=(p['x'], p['f']), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', color=c)

    leg = [
        Line2D([0], [0], marker='o', color='w', label='E domain', markerfacecolor=cols['E'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='G domain', markerfacecolor=cols['G'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='A domain', markerfacecolor=cols['A'], markersize=10),
        Line2D([0], [0], color=red, lw=2, label='Aap'),
    ]
    ax.legend(handles=leg, loc='upper left', fontsize=11)

    ax.set_title("Aap (E=275pN, G=550pN, A=950pN)", fontsize=15, fontweight='bold')
    ax.set_xlabel("Extension (nm)", fontsize=12)
    ax.set_ylabel("Force (pN)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(xa) * 1.05)
    ax.set_ylim(-50, max(fa_curve) * 1.2)

    plt.tight_layout()
    save(fig, "fig02_aap_only")
    plt.show()
    return pk_aap


# =============================================================================
# VERIFICATION  -  check the peaks against the expected values
# =============================================================================
def verify():
    print("=" * 74)
    print("WLC verification  (model output vs expected)")
    print("=" * 74)
    rows = [
        ("Fig 1  E  rupture force (pN)",   wlc(0.902, 1.0), 273.5),
        ("Fig 1  G5 rupture force (pN)",   wlc(0.931, 1.0), 545.2),
        ("Fig 1  SasG A rupture force (pN)", wlc(0.938, 1.0), 673.7),
        ("Fig 1  Aap A rupture force (pN)",  wlc(0.948, 1.0), 954.8),
        ("Fig 1  SasG max extension (nm)", 333.4 * 0.938, 312.7),
        ("Fig 1  Aap max extension (nm)",  405.4 * 0.948, 384.3),
    ]
    for lab, got, doc in rows:
        flag = "ok " if abs(got - doc) <= 0.1 else "CHECK"
        print(f"  {lab:34s} {got:9.2f}   expected {doc:8.1f}   {flag}")

    _, _, pk = sawtooth(le, lg, la, fe, fg, fa)
    for name, target in (("E1", fe), ("G1", fg), ("A", fa)):
        f = next(p['f'] for p in pk if p['label'] == name)
        flag = "ok " if target <= f <= target * 1.02 else "CHECK"
        print(f"  Fig 2  {name:<3s} peak (pN)               {f:9.2f}   target   {target:8.1f}   {flag}")
    print("=" * 74)


# =============================================================================
# MAIN
# =============================================================================
def main():
    verify()
    print()
    fig01()
    fig02()
    print("\nAll figures written to", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
