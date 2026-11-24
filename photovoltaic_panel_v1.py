# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:55:05 2026

@author: VK
"""

import math
import matplotlib.pyplot as plt

# -----------------------------
# Given data
# -----------------------------
G = 700.0                        # solar irradiation [W/m^2]
rho_reflect = 0.07               # reflected from glass top surface
abs_glass_top = 0.10             # absorbed at glass top surface
abs_si = 0.83                    # transmitted and absorbed in silicon

a = 0.553
b = 0.001                        # [1/K], T in Kelvin

t_glass = 3e-3                   # [m]
k_glass = 1.4                    # [W/m-K]

t_adh = 0.1e-3                   # [m]
k_adh = 145.0                    # [W/m-K]

# Layers below silicon (included for completeness, but bottom is insulated -> no steady 1D downward heat flux)
t_solder = 0.1e-3
t_aln = 2e-3

h = 35.0                         # [W/m^2-K]
eps = 0.9
sigma = 5.670374419e-8           # [W/m^2-K^4]

T_inf = 20.0 + 273.15            # air temperature [K]
T_sur = 20.0 + 273.15            # surroundings "outside wall" temperature [K]

L = 1.0                          # panel length [m]
W = 0.1                          # panel width [m]
A = L * W                        # area [m^2]

# Heat inputs
q_glass = abs_glass_top * G      # heat deposited at outer glass surface [W/m^2]
G_si = abs_si * G                # solar absorbed in silicon [W/m^2]

# Conduction resistance from silicon plane up to the outer glass surface
R_up = t_adh / k_adh + t_glass / k_glass  # [m^2-K/W]

# From rearranging:
#   eff = a - b*Tsi
#   Tsi = Ts + q_si*R_up
#   q_si = (1-eff)*G_si
# -> q_si can be written explicitly as a function of Ts:
den = 1.0 - b * R_up * G_si
if den <= 0:
    raise ValueError("Unphysical parameters: denominator <= 0. Check b, R_up, G_si.")

def q_si_from_Ts(Ts):
    """Thermal power generated in silicon per area, as a function of outer surface temperature Ts."""
    return ((1.0 - a + b * Ts) * G_si) / den

def eff_Tsi_from_Ts(Ts):
    q_si = q_si_from_Ts(Ts)
    T_si = Ts + q_si * R_up
    eff = a - b * T_si
    # Clamp to [0,1] physically
    eff = max(0.0, min(1.0, eff))
    return eff, T_si, q_si

def surface_balance(Ts):
    """
    f(Ts)=0 steady energy balance at outer surface:
      convection + radiation = q_glass + q_si
    """
    q_si = q_si_from_Ts(Ts)
    q_conv = h * (Ts - T_inf)
    q_rad = eps * sigma * (Ts**4 - T_sur**4)
    return (q_conv + q_rad) - (q_glass + q_si)

# -----------------------------
# Solve for Ts via bisection
# -----------------------------
T_lo = T_inf          # near ambient
T_hi = 500.0          # K (upper bracket)
f_lo = surface_balance(T_lo)
f_hi = surface_balance(T_hi)

# Expand bracket if needed
while f_lo * f_hi > 0 and T_hi < 1200:
    T_hi += 200
    f_hi = surface_balance(T_hi)

if f_lo * f_hi > 0:
    raise RuntimeError("Failed to bracket the surface temperature root.")

for _ in range(120):
    T_mid = 0.5 * (T_lo + T_hi)
    f_mid = surface_balance(T_mid)
    if f_lo * f_mid <= 0:
        T_hi, f_hi = T_mid, f_mid
    else:
        T_lo, f_lo = T_mid, f_mid

Ts = 0.5 * (T_lo + T_hi)

eff, T_si, q_si = eff_Tsi_from_Ts(Ts)
q_elec = eff * G_si              # electrical power density [W/m^2]
P_elec = q_elec * A              # electrical power [W]

# -----------------------------
# Print key results
# -----------------------------
print("=== Steady-state results ===")
print(f"Panel area A = {A:.3f} m^2")
print(f"Outer glass surface temperature Ts = {Ts-273.15:.2f} °C  ({Ts:.2f} K)")
print(f"Silicon temperature T_si = {T_si-273.15:.2f} °C  ({T_si:.2f} K)")
print(f"Efficiency eta = {eff:.4f}  [-]")
print(f"Electrical power density = {q_elec:.2f} W/m^2")
print(f"Electrical power (1 m x 0.1 m) = {P_elec:.2f} W")

# -----------------------------
# Create schematic picture
# -----------------------------
layers = [
    ("Air (20°C)", 0.8, "#f0f0f0"),
    ("Glass\n3 mm\nk=1.4", 1.3, "#cfe8ff"),
    ("Adhesive\n0.1 mm\nk=145", 0.5, "#ffe6b3"),
    ("Si (very thin)\nHeat gen + elec", 0.45, "#d6f5d6"),
    ("Solder\n0.1 mm", 0.5, "#ffd6d6"),
    ("AlN substrate\n2 mm", 1.0, "#e6e6ff"),
    ("Insulated bottom", 0.8, "#f0f0f0"),
]

fig, ax = plt.subplots(figsize=(7.5, 5))
y = 0
width = 4.0

for name, hgt, col in layers[::-1]:
    ax.add_patch(plt.Rectangle((0, y), width, hgt, facecolor=col, edgecolor="black", linewidth=1))
    ax.text(width/2, y + hgt/2, name, ha="center", va="center", fontsize=10)
    y += hgt

ax.set_xlim(-1.4, width + 1.4)
ax.set_ylim(0, y)
ax.axis("off")

# Heat flux arrow out of top surface
ax.annotate("", xy=(width/2, y-0.2), xytext=(width/2, y+0.9),
            arrowprops=dict(arrowstyle="->", linewidth=2))
ax.text(width/2 + 0.1, y+0.85, "q_out = h(Ts-Ta) + εσ(Ts⁴-Tsur⁴)",
        ha="left", va="top", fontsize=9)

# Solar arrow into top
ax.annotate("", xy=(width/2, y-0.9), xytext=(width/2, y+0.1),
            arrowprops=dict(arrowstyle="->", linewidth=2))
ax.text(width/2 + 0.1, y-0.7, "Solar G = 700 W/m²\n7% refl, 10% absorbed @ top\n83% absorbed in Si",
        ha="left", va="center", fontsize=9)

txt = (f"Computed steady state:\n"
       f"Ts = {Ts-273.15:.2f} °C\n"
       f"Tsi = {T_si-273.15:.2f} °C\n"
       f"η = {eff:.4f}\n"
       f"P_elec = {P_elec:.2f} W (A={A:.2f} m²)")
ax.text(-1.25, y*0.5, txt, ha="left", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"))

out_path = "pv_panel_schematic.png"
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSchematic saved to: {out_path}")