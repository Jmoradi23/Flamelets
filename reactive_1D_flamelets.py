#!/usr/bin/env python3
"""
FAST + STABLE Ember initialization from CSV for H2/O2 diffusion flame.
Uses actual CSV x-grid and values. No arbitrary rescaling.
Each .h5 profile is saved as a corresponding .csv.
Plots T-x evolution.
"""

import os
import numpy as np
import pandas as pd
from ember import *
import cantera as ct
from scipy.ndimage import gaussian_filter1d
import h5py
import matplotlib.pyplot as plt
import glob

# -------------------------
# Settings
# -------------------------
mechanism_file = "h2-burke.yaml"
input_csv = "inert_profile.csv"       # CSV from FlameMaster/Ember
output_dir = "run/ember_fast"
os.makedirs(output_dir, exist_ok=True)

smoothing_sigma = 2                   # Gaussian smoothing
initial_dt = 5e-10
min_species = 1e-18
radicals = ['H', 'O', 'OH', 'HO2', 'H2O2']
T_min, T_max = 200.0, 4500.0

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv(input_csv, sep=r"\s+|,", engine="python")

required_cols = ['x', 'T']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV must have column '{col}'")

x = df['x'].to_numpy(float)
T = df['T'].to_numpy(float)

# Reverse if x decreases
if x[0] > x[-1]:
    x = x[::-1]
    df = df.iloc[::-1].reset_index(drop=True)
    T = T[::-1]

# Species columns
species_cols = [c for c in df.columns if c.startswith("Y_")]
species_names = [c[2:] for c in species_cols]

print(f"Loaded {len(x)} points from CSV.")
print("Species columns found in CSV:")
for col, sp in zip(species_cols, species_names):
    print(f"  {col} → {sp}")

# Stack Y values
Y_raw = np.vstack([df[c].to_numpy(float) for c in species_cols]).T  # (nx, n_species)

# -------------------------
# Reorder species to match mechanism
# -------------------------
gas = ct.Solution(mechanism_file)
mech_species = gas.species_names

Y_reordered = np.zeros((len(x), len(mech_species)), dtype=float)
for i, sp in enumerate(mech_species):
    if sp in species_names:
        Y_reordered[:, i] = Y_raw[:, species_names.index(sp)]
    else:
        Y_reordered[:, i] = min_species

# Apply species floor
for i, sp in enumerate(mech_species):
    if sp in radicals:
        Y_reordered[:, i] = np.maximum(Y_reordered[:, i], min_species)
Y_reordered = np.maximum(Y_reordered, min_species)

# Normalize rows
Y_sum = Y_reordered.sum(axis=1)
Y_sum[Y_sum == 0] = 1.0
Y_reordered = (Y_reordered.T / Y_sum).T

# -------------------------
# Smoothing
# -------------------------
if smoothing_sigma > 0:
    T = gaussian_filter1d(T, smoothing_sigma)
    Y_reordered = gaussian_filter1d(Y_reordered, smoothing_sigma, axis=0)
T = np.clip(T, T_min, T_max)

# Store initial condition for plotting
T_initial_for_plot = T.copy()
Y_initial_for_plot = Y_reordered.copy()

# -------------------------
# Ember configuration
# -------------------------
conf = Config(
    Paths(outputDir=output_dir),

    Chemistry(
        mechanismFile=mechanism_file
    ),

    General(
        flameGeometry="cylindrical",
        nThreads=8,
        fixedLeftLocation=True,
        chemistryIntegrator="qss"
    ),

    InitialCondition(
        restartFile='outputs/profNow_inert.h5'  
    ),

    StrainParameters(initial=100, final=100),

    TerminationCondition(
        tEnd=2e-6
    ),

    Times(
        globalTimestep=initial_dt,
        profileStepInterval=2000,
        regridStepInterval=5000
    )
)

# -------------------------
# Run Ember
# -------------------------
try:
    conf.run()
    print(f"Finished Ember (fast mode). Output in: {output_dir}")
except Exception as e:
    print("Ember failed:")
    print(e)

# -------------------------
# Post-processing: export each .h5 profile to .csv
# -------------------------
prof_files = sorted(glob.glob(os.path.join(output_dir, "prof*.h5")))

if prof_files:
    print("\nExporting .h5 profiles to CSV files...")
    for prof_file in prof_files:
        with h5py.File(prof_file, 'r') as f:
            x_prof = f['x'][:]
            T_prof = f['T'][:]
            Y_prof = f['Y'][:]

        # Ensure shape consistency
        if Y_prof.shape[0] == len(x_prof) and Y_prof.shape[1] == len(mech_species):
            Y_to_save = Y_prof
        elif Y_prof.shape[1] == len(x_prof) and Y_prof.shape[0] == len(mech_species):
            Y_to_save = Y_prof.T
        else:
            raise ValueError(f"Unexpected Y_prof shape: {Y_prof.shape}, x_prof: {len(x_prof)}")

        df_prof = pd.DataFrame({'x': x_prof, 'T': T_prof})
        for j, sp in enumerate(mech_species):
            df_prof[f"Y_{sp}"] = Y_to_save[:, j]

        csv_file = os.path.splitext(prof_file)[0] + ".csv"
        df_prof.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

    # -------------------------
    # Plot T-x evolution
    # -------------------------
    plt.figure(figsize=(7,5))
    # Original CSV
    plt.plot(x, T_initial_for_plot, '--', label='Initial CSV (t=0)')

    # t=0 Ember internal grid (first profile)
    df_first = pd.read_csv(os.path.splitext(prof_files[0])[0] + ".csv")
    plt.plot(df_first['x'], df_first['T'], '-', label='t=0 Ember grid')

    # Middle and end profiles
    mid_idx = len(prof_files)//2
    df_mid = pd.read_csv(os.path.splitext(prof_files[mid_idx])[0] + ".csv")
    plt.plot(df_mid['x'], df_mid['T'], '-', label='Middle profile')
    df_end = pd.read_csv(os.path.splitext(prof_files[-1])[0] + ".csv")
    plt.plot(df_end['x'], df_end['T'], '-', label='End profile')

    plt.xlabel('x [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature profile evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"No profile files found in {output_dir}, skipping CSV export and plotting.")

# -------------------------
# Print x, T, and Y at t=0
# -------------------------
print("\nFinal domain initialization (Ember internal grid at t=0):")
for i in range(len(T_initial_for_plot)):
    species_str = ", ".join([f"{sp}:{Y_initial_for_plot[i,j]:.3e}" for j, sp in enumerate(mech_species)])
    print(f"x={x[i]:.6e}, T={T_initial_for_plot[i]:.2f}, {species_str}")
