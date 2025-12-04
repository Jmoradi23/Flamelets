#!/usr/bin/env python3
"""
Ember-compatible inert counterflow flame generator:
- Chemistry disabled
- T(x), Y(x)
- U initialized to zero for strain-rate-independent restart
- V set to zero
- CSV + HDF5 output
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import cantera as ct

# -------------------------
# USER SETTINGS
# -------------------------
mechanism_file = "h2-burke.yaml"

width = 0.06
fuel_X = "H2:1"
fuel_T = 300.0
fuel_mdot = 0.15

ox_X = "O2:0.12, AR:0.88"
ox_T = 1300.0
ox_mdot = 1.0

P = 1e5

smoothing_sigma_T = 4
smoothing_sigma_Y = 4
n_resample_full = 300
n_resample_zoom = 300

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Disable chemistry
# -------------------------
def disable_reactions_cantera(gas):
    try:
        gas.set_multiplier(0.0)
        return True
    except Exception:
        pass
    try:
        for i in range(gas.n_reactions):
            gas.set_multiplier(i, 0.0)
        return True
    except Exception:
        pass
    return False

gas = ct.Solution(mechanism_file)
disable_reactions_cantera(gas)

# -------------------------
# Counterflow flame setup
# -------------------------
f = ct.CounterflowDiffusionFlame(gas, width=width)
f.soret_enabled = False
f.fuel_inlet.X = fuel_X
f.fuel_inlet.T = fuel_T
f.fuel_inlet.mdot = fuel_mdot
f.oxidizer_inlet.X = ox_X
f.oxidizer_inlet.T = ox_T
f.oxidizer_inlet.mdot = ox_mdot
f.P = P
f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02)
f.max_grid_points = 8000
f.solve(loglevel=0, auto=True)

# -------------------------
# Extract fields
# -------------------------
x_grid = np.array(f.grid)
T = np.array(f.T)
species = gas.species_names
Y_full = np.array(f.Y)
if Y_full.shape[0] == len(species):
    Y = Y_full.T
else:
    Y = Y_full

# Smooth and normalize
T_s = gaussian_filter1d(T, smoothing_sigma_T)
Y_s = gaussian_filter1d(Y, smoothing_sigma_Y, axis=0)
Y_s = (Y_s.T / Y_s.sum(axis=1)).T

# -------------------------
# Adaptive resampling
# -------------------------
Z = np.linspace(0, 1, n_resample_full)  # just uniform pseudo-Z
x_full = np.interp(Z, np.linspace(0, 1, len(x_grid)), x_grid)
T_full = np.interp(Z, np.linspace(0, 1, len(T_s)), T_s)
Y_full_resamp = np.zeros((n_resample_full, len(species)))
for j in range(len(species)):
    Y_full_resamp[:, j] = np.interp(Z, np.linspace(0, 1, len(Y_s)), Y_s[:, j])

# -------------------------
# U and V initialization
# -------------------------
U_full = np.zeros_like(x_full)  # Ember will compute proper U
V_full = np.zeros_like(x_full)

P_full = np.full_like(x_full, P)

# -------------------------
# Save Ember-compatible HDF5
# -------------------------
with h5py.File(os.path.join(output_dir,"profNow_inert.h5"), "w") as h5:
    h5.create_dataset("x", data=x_full)
    h5.create_dataset("T", data=T_full)
    h5.create_dataset("Y", data=Y_full_resamp)
    h5.create_dataset("U", data=U_full)
    h5.create_dataset("V", data=V_full)
    h5.attrs["species"] = np.array(species, dtype=h5py.string_dtype())

print(f"\n✅ Ember-compatible HDF5 saved in '{output_dir}/profNow_inert.h5' ✅")
