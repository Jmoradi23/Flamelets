#!/usr/bin/env python3
"""
Reactive evolution over full Z-space with plotting restricted to
uniform Z region (0 ≤ Z ≤ 0.2). Includes inert mixing overlay.
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# --- High precision dtype ---
dtype = np.float128 if hasattr(np, "float128") else np.float64

# --- Load inert counterflow solution ---
data = np.load('counterflow_inert_solution.npz')
z_grid = data['z']
P_grid = data['P']
T_grid = data['T']
Y_H2_grid = data['Y_H2']
Y_O2_grid = data['Y_O2']
Y_AR_grid = data['Y_AR']
Z_grid = data['Z']

# --- Check grid coverage ---
print(f"Loaded {len(Z_grid)} mixture fraction points.")
print(f"Z range: {Z_grid.min():.6f} → {Z_grid.max():.6f}")

# --- Cantera gas ---
gas = ct.Solution('Konnov2008.yaml')

# --- Simulation parameters ---
time_end = 0.001       # 1 ms
dt_output = 1e-8       # 10 ns
time_array = np.arange(0, time_end + dt_output, dt_output)

# --- Prepare arrays for full Z coverage ---
nZ = len(Z_grid)
nT = len(time_array)
T_data = np.zeros((nT, nZ))
Y_H2_data = np.zeros((nT, nZ))
Y_O2_data = np.zeros((nT, nZ))
Y_AR_data = np.zeros((nT, nZ))
P_data = np.zeros((nT, nZ))

# --- Compute Zst from reference states ---
MH = dtype(gas.atomic_weight('H'))
MO = dtype(gas.atomic_weight('O'))

gas.X = 'H2:1'
YH1, YO1 = dtype(gas.elemental_mass_fraction('H')), dtype(gas.elemental_mass_fraction('O'))
gas.X = "O2:0.082, AR:0.918"
YH2, YO2 = dtype(gas.elemental_mass_fraction('H')), dtype(gas.elemental_mass_fraction('O'))

num_st = 0.5*(0.0 - YH2)/MH - (0.0 - YO2)/MO
den_st = 0.5*(YH1 - YH2)/MH - (YO1 - YO2)/MO
Zst = num_st / (den_st + dtype(1e-30))
print(f"Stoichiometric mixture fraction Zst ≈ {Zst:.6f}")

# --- Loop over each Z in full range ---
for j in range(nZ):
    gas.TPY = T_grid[j], P_grid[j], {'H2': Y_H2_grid[j], 'O2': Y_O2_grid[j], 'AR': Y_AR_grid[j]}
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    
    for k, t in enumerate(time_array):
        sim.advance(t)
        T_data[k, j] = reactor.T
        P_data[k, j] = reactor.thermo.P
        Y_H2_data[k, j] = reactor.Y[gas.species_index('H2')]
        Y_O2_data[k, j] = reactor.Y[gas.species_index('O2')]
        Y_AR_data[k, j] = reactor.Y[gas.species_index('AR')]

print("✅ Simulation completed over full Z-space (no truncation).")

# --- Save full results ---
np.savez('HR_full_time_Z_data_full_range.npz',
         time=time_array,
         Z=Z_grid,
         T=T_data,
         P=P_data,
         Y_H2=Y_H2_data,
         Y_O2=Y_O2_data,
         Y_AR=Y_AR_data)
print("✅ Full data saved to 'HR_full_time_Z_data_full_range.npz'.")

# ==========================================================
# ===   RESTRICTED Z REGION FOR PLOTTING (0 ≤ Z ≤ 0.2)   ===
# ==========================================================
Z_min, Z_max = 0.0, 0.2
bound_indices = np.where((Z_grid >= Z_min) & (Z_grid <= Z_max))[0]
Z_bound = Z_grid[bound_indices]

print(f"Plotting restricted to Z range: {Z_min:.3f} ≤ Z ≤ {Z_max:.3f} "
      f"({len(Z_bound)} points retained)")

# --- Plot Temperature vs Z (restricted range) ---
plt.figure(figsize=(9,6))
time_indices = np.linspace(0, len(time_array)-1, 6, dtype=int)
for idx_t in time_indices:
    plt.plot(Z_bound, T_data[idx_t, bound_indices], label=f"t={time_array[idx_t]*1000:.3f} ms")
plt.plot(Z_bound, T_grid[bound_indices], 'k--', lw=2, label="Initial mixing (inert)")
plt.axvline(Zst, color='r', linestyle=':', lw=2, label=f"Zst ≈ {Zst:.6f}")
plt.xlabel("Mixture fraction Z")
plt.ylabel("Temperature [K]")
plt.title("Temperature vs Z (0 ≤ Z ≤ 0.2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T_vs_Z_0_0.2.png", dpi=300)
plt.show()

# --- Plot Pressure vs Z (restricted range) ---
plt.figure(figsize=(9,6))
for idx_t in time_indices:
    plt.plot(Z_bound, P_data[idx_t, bound_indices]/1e5, label=f"t={time_array[idx_t]*1000:.3f} ms")
plt.plot(Z_bound, P_grid[bound_indices]/1e5, 'k--', lw=2, label="Initial mixing (inert)")
plt.axvline(Zst, color='r', linestyle=':', lw=2, label=f"Zst ≈ {Zst:.6f}")
plt.xlabel("Mixture fraction Z")
plt.ylabel("Pressure [bar]")
plt.title("Pressure vs Z (0 ≤ Z ≤ 0.2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("P_vs_Z_0_0.2.png", dpi=300)
plt.show()

# --- Plot Species vs Z (restricted range) ---
plt.figure(figsize=(10,6))
for idx_t in time_indices:
    plt.plot(Z_bound, Y_H2_data[idx_t, bound_indices], label=f"H2 t={time_array[idx_t]*1000:.3f} ms")
    plt.plot(Z_bound, Y_O2_data[idx_t, bound_indices], '--', label=f"O2 t={time_array[idx_t]*1000:.3f} ms")
    plt.plot(Z_bound, Y_AR_data[idx_t, bound_indices], ':', label=f"AR t={time_array[idx_t]*1000:.3f} ms")

# Inert initial profiles
plt.plot(Z_bound, Y_H2_grid[bound_indices], 'k-', lw=2, label="Initial H2")
plt.plot(Z_bound, Y_O2_grid[bound_indices], 'k--', lw=2, label="Initial O2")
plt.plot(Z_bound, Y_AR_grid[bound_indices], 'k:', lw=2, label="Initial AR")

plt.xlabel("Mixture fraction Z")
plt.ylabel("Mass fraction")
plt.title("Species mass fractions vs Z (0 ≤ Z ≤ 0.2)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("Species_vs_Z_0_0.2.png", dpi=300)
plt.show()
