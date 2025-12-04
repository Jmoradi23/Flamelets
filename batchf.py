import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# --- high precision dtype ---
dtype = np.float128 if hasattr(np, "float128") else np.float64

# --- Load inert counterflow solution ---
data = np.load('counterflow_inert_solution.npz')
z_grid = np.array(data['z'], dtype=float)   # original adaptive flame grid (may be non-uniform)
P_grid = np.array(data['P'], dtype=float)
T_grid = np.array(data['T'], dtype=float)
Y_H2_grid = np.array(data['Y_H2'], dtype=float)
Y_O2_grid = np.array(data['Y_O2'], dtype=float)
Y_AR_grid = np.array(data['Y_AR'], dtype=float)
Z_grid = np.array(data['Z'], dtype=float)   # mixture fraction on adaptive grid

# --- sanity: sort by Z (required for interpolation) and remove duplicates if any ---
order = np.argsort(Z_grid)
Z_grid = Z_grid[order]
T_grid = T_grid[order]
P_grid = P_grid[order]
Y_H2_grid = Y_H2_grid[order]
Y_O2_grid = Y_O2_grid[order]
Y_AR_grid = Y_AR_grid[order]

# remove exact-duplicate Z entries (if any)
_, uniq_idx = np.unique(Z_grid, return_index=True)
Z_grid = Z_grid[uniq_idx]
T_grid = T_grid[uniq_idx]
P_grid = P_grid[uniq_idx]
Y_H2_grid = Y_H2_grid[uniq_idx]
Y_O2_grid = Y_O2_grid[uniq_idx]
Y_AR_grid = Y_AR_grid[uniq_idx]

# --- Define uniform Z grid covering full 0..1 ---
N_uniform = max(len(Z_grid), 601)     # choose at least 601 points (adjust if you want)
Z_uniform = np.linspace(0.0, 1.0, N_uniform)

# --- Interpolate inert solution onto uniform Z ---
# use numpy.interp (1D linear). Provide left/right fill using endpoint values.
T_uniform = np.interp(Z_uniform, Z_grid, T_grid, left=T_grid[0], right=T_grid[-1])
P_uniform = np.interp(Z_uniform, Z_grid, P_grid, left=P_grid[0], right=P_grid[-1])
Y_H2_uniform = np.interp(Z_uniform, Z_grid, Y_H2_grid, left=Y_H2_grid[0], right=Y_H2_grid[-1])
Y_O2_uniform = np.interp(Z_uniform, Z_grid, Y_O2_grid, left=Y_O2_grid[0], right=Y_O2_grid[-1])
Y_AR_uniform = np.interp(Z_uniform, Z_grid, Y_AR_grid, left=Y_AR_grid[0], right=Y_AR_grid[-1])

# --- Cantera gas ---
gas = ct.Solution('Konnov2008.yaml')

# --- Simulation parameters (unchanged) ---
time_end = 0.001       # 1 ms
dt_output = 1e-8       # 10 ns
time_array = np.arange(0, time_end + dt_output, dt_output)

# --- Prepare arrays to store results (time x Z) ---
nt = len(time_array)
nz = len(Z_uniform)
T_data = np.zeros((nt, nz), dtype=float)
Y_H2_data = np.zeros_like(T_data)
Y_O2_data = np.zeros_like(T_data)
Y_AR_data = np.zeros_like(T_data)

# --- Compute Zst from reference states (keeps dtype usage like your file) ---
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

# --- Loop over the uniform Z grid: initialize reactors from interpolated initial state ---
# Note: this loop may be heavy if time_array is large; we keep your original time discretization.
for idx in range(nz):
    # initialize gas using mass fractions (the saved arrays are mass fractions)
    Ydict = {'H2': float(Y_H2_uniform[idx]),
             'O2': float(Y_O2_uniform[idx]),
             'AR': float(Y_AR_uniform[idx])}
    # TPY: temperature (K), pressure (Pa), mass fractions dict
    gas.TPY = float(T_uniform[idx]), float(P_uniform[idx]), Ydict

    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])

    # advance in time and record
    for k, t in enumerate(time_array):
        sim.advance(t)
        T_data[k, idx] = reactor.T
        Y_H2_data[k, idx] = reactor.Y[gas.species_index('H2')] if 'H2' in gas.species_names else 0.0
        Y_O2_data[k, idx] = reactor.Y[gas.species_index('O2')] if 'O2' in gas.species_names else 0.0
        Y_AR_data[k, idx] = reactor.Y[gas.species_index('AR')] if 'AR' in gas.species_names else 0.0

print("Simulation completed for uniform Z-space (0 ≤ Z ≤ 1).")

# --- Save full time × Z data ---
np.savez('HR_full_time_Z_data_full_range.npz',
         time=time_array,
         Z=Z_uniform,
         T=T_data,
         Y_H2=Y_H2_data,
         Y_O2=Y_O2_data,
         Y_AR=Y_AR_data)

print("Full time × Z data saved to 'HR_full_time_Z_data_full_range.npz'.")

# --- Plot: show initial interpolated inert profile vs reactor-evolved for selected times ---
plt.figure(figsize=(8,6))
time_indices = np.linspace(0, nt-1, 6, dtype=int)  # choose 6 times including start/end
# plot initial inert (interpolated) temperature
plt.plot(Z_uniform, T_uniform, 'k--', lw=2, label="Initial inert T(Z) (interp)")
# plot reactor results at selected times
for idx_t in time_indices:
    plt.plot(Z_uniform, T_data[idx_t, :], label=f"t={time_array[idx_t]*1000:.3f} ms")
plt.axvline(float(Zst), color='k', linestyle=':', lw=1.5, label=f"Zst ≈ {Zst:.6f}")
plt.xlabel("Mixture fraction Z")
plt.ylabel("Temperature [K]")
plt.title("Temperature vs Z: initial inert (interp) and reactor evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T_vs_Z_full_range_uniform.png", dpi=300)
plt.show()

# --- Plot species: initial inert vs reactor for selected times ---
plt.figure(figsize=(10,6))
# initial species lines (interpolated)
plt.plot(Z_uniform, Y_H2_uniform, 'k--', lw=1.8, label="Initial inert Y_H2 (interp)")
plt.plot(Z_uniform, Y_O2_uniform,  'k-.', lw=1.8, label="Initial inert Y_O2 (interp)")
plt.plot(Z_uniform, Y_AR_uniform,  'k:',  lw=1.8, label="Initial inert Y_AR (interp)")

# reactor results at selected times (overlay)
for idx_t in time_indices:
    plt.plot(Z_uniform, Y_H2_data[idx_t, :], label=f"H2 t={time_array[idx_t]*1000:.3f} ms")
    plt.plot(Z_uniform, Y_O2_data[idx_t, :], '--', label=f"O2 t={time_array[idx_t]*1000:.3f} ms")
    plt.plot(Z_uniform, Y_AR_data[idx_t, :], ':', label=f"AR t={time_array[idx_t]*1000:.3f} ms")

plt.xlabel("Mixture fraction Z")
plt.ylabel("Mass fraction")
plt.title("Species mass fractions vs Z: initial inert (interp) and reactor evolution")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("Species_vs_Z_full_range_uniform.png", dpi=300)
plt.show()
