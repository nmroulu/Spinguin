"""
This is an advanced example of a SABRE (Signal Amplification by Reversible Exchange) simulation of pyridine.
The spin system consists of hydride protons and one pyridine ligand.
Chemical exchange and relaxation processes are taken into account.

This simulation may take approximately 10-15 minutes to run on a laptop with an 11th-generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from spinguin import (
    SpinSystem, hamiltonian, relaxation, propagator, unit_state, singlet, measure,
    associate, dissociate, permute_spins, ZQ_basis_map, ZQ_filter
)

# Simulation settings
max_spin_order = 4  # Maximum spin order to consider in the simulation
B0 = 7e-3  # Magnetic field strength in Tesla
dt = 2e-3  # Time step for the simulation in seconds
N_steps = 30000  # Number of simulation steps
k_H2 = 1.6  # Exchange rate for H2 (in arbitrary units)
k_s = 10  # Exchange rate for the substrate (in arbitrary units)
tau_c_c = 43e-12  # Correlation time for the complex in seconds
tau_c_s = 5.7e-12  # Correlation time for the substrate in seconds
c_c = 0.0005  # Concentration of the complex (in mol/L)
c_s = 0.015  # Concentration of the substrate (in mol/L)

# Assign isotopes in the three spin systems
isotopes_c = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])  # Complex
isotopes_s = np.array(['1H', '1H', '1H', '1H', '1H', '14N'])  # Substrate
isotopes_H2 = np.array(['1H', '1H'])  # H2 molecule

# Define spin maps for the exchange processes
spin_map_c = (0, 1, 2, 3, 4, 5, 6, 7)  # Spin map for the complex
spin_map_s = (2, 3, 4, 5, 6, 7)  # Spin map for the substrate
spin_map_H2 = (0, 1)  # Spin map for H2

# # Define a rotation map for pyridine that accounts for symmetry
# rot_map = (1, 0, 3, 2, 4, 5)  # Symmetry rotation map for pyridine
# Define a permutation map for pyridine that accounts for symmetry
perm_map = (0, 1, 2, 3, 4, 5) # NOTE: Perttu's edit

# Assign chemical shifts (in ppm)
chemical_shifts_c = np.array([-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77, 43.60])  # Complex
chemical_shifts_s = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])  # Substrate
chemical_shifts_H2 = np.array([4.2, 4.2])  # H2 molecule

# Assign scalar couplings (in Hz)
J_couplings_c = np.array([
    # Scalar coupling matrix for the complex
    [ 0,     0,      0,      0,      0,      0,      0,     0],
    [-6.53,  0,      0,      0,      0,      0,      0,     0],
    [ 0.00,  1.66,   0,      0,      0,      0,      0,     0],
    [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0],					
    [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0,     0],					
    [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0,     0],			
    [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0,     0],
    [-0.30,  15.91,  4.47,   0.04,   1.79,   0,     -0.46,  0]
])
J_couplings_s = np.array([
    # Scalar coupling matrix for the substrate
    [ 0,     0,      0,      0,      0,      0],
    [-1.04,  0,      0,      0,      0,      0],
    [ 4.85,  1.05,   0,      0,      0,      0],
    [ 1.05,  4.85,   0.71,   0,      0,      0],
    [ 1.24,  1.24,   7.55,   7.55,   0,      0],
    [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
])

# Assign coordinates (in Angstroms)
xyz_c = np.array([
    # Cartesian coordinates for the complex
    [ 0.9649170,  1.2271534, -1.2031835],
    [ 1.9547078, -0.4342818, -0.1922623],
    [-2.5492743, -0.9988969, -0.2721286],
    [-1.3895773, -2.6746310, -1.2331514],
    [-4.5704762, -1.0068808, -1.6740361],
    [-1.5724294, -5.0099155, -0.4837575],
    [-4.5894708,  0.3577455, -3.7835019],
    [-1.4702745,  0.2327446, -1.5095832]
])
xyz_s = np.array([
    # Cartesian coordinates for the substrate
    [ 2.0495335, 0.0000000, -1.4916842],
    [-2.0495335, 0.0000000, -1.4916842],
    [ 2.1458878, 0.0000000,  0.9846086],
    [-2.1458878, 0.0000000,  0.9846086],
    [ 0.0000000, 0.0000000,  2.2681296],
    [ 0.0000000, 0.0000000, -1.5987077]
])

# Assign shielding tensors (in ppm)
shielding_c = np.zeros((8, 3, 3))  # Shielding tensors for the complex
shielding_c[7] = np.array([
    [-134.70, -123.93, -49.86],
    [-147.79,  64.47,   221.96],
    [-62.63,   223.57, -60.57]
])
shielding_s = np.zeros((6, 3, 3))  # Shielding tensors for the substrate
shielding_s[5] = np.array([
    [-406.20, 0.00,   0.00],
    [ 0.00,   299.44, 0.00],
    [ 0.00,   0.00,  -181.07]
])

# Assign EFG tensors (in atomic units)
efg_c = np.zeros((8, 3, 3))  # Electric field gradient tensors for the complex
efg_c[7] = np.array([
    [-0.3426, -0.0417, -0.4514],
    [-0.0417,  0.3727,  0.1186],
    [-0.4514,  0.1186, -0.0301]
])
efg_s = np.zeros((6, 3, 3))  # Electric field gradient tensors for the substrate
efg_s[5] = np.array([
    [0.3069, 0.0000,  0.0000],
    [0.0000, 0.7969,  0.0000],
    [0.0000, 0.0000, -1.1037]
])

# Initialize the spin systems
spin_system_c = SpinSystem(isotopes_c, chemical_shifts_c, J_couplings_c, xyz_c, shielding_c, efg_c, max_spin_order)
spin_system_s = SpinSystem(isotopes_s, chemical_shifts_s, J_couplings_s, xyz_s, shielding_s, efg_s, max_spin_order)
spin_system_H2 = SpinSystem(isotopes_H2)

# Create the Hamiltonians
H_c = hamiltonian(spin_system_c, B0)  # Hamiltonian for the complex
H_s = hamiltonian(spin_system_s, B0)  # Hamiltonian for the substrate

# Create the relaxation superoperators
R_c = relaxation(spin_system_c, H_c, B0, tau_c_c, include_sr2k=True)  # Relaxation for the complex
R_s = relaxation(spin_system_s, H_s, B0, tau_c_s, include_sr2k=True)  # Relaxation for the substrate

# Switch to the zero-quantum (ZQ) basis
ZQ_map_c = ZQ_basis_map(spin_system_c)  # ZQ basis map for the complex
ZQ_map_s = ZQ_basis_map(spin_system_s)  # ZQ basis map for the substrate
ZQ_basis_map(spin_system_H2)  # ZQ basis map for H2

# Transform the Hamiltonians and relaxation superoperators to the ZQ basis
H_c = ZQ_filter(spin_system_c, H_c, ZQ_map_c)  # Filter Hamiltonian for the complex
H_s = ZQ_filter(spin_system_s, H_s, ZQ_map_s)  # Filter Hamiltonian for the substrate
R_c = ZQ_filter(spin_system_c, R_c, ZQ_map_c)  # Filter relaxation for the complex
R_s = ZQ_filter(spin_system_s, R_s, ZQ_map_s)  # Filter relaxation for the substrate

# Create the time propagators
P_c = propagator(dt, H_c, R_c)  # Time propagator for the complex
P_s = propagator(dt, H_s, R_s)  # Time propagator for the substrate

# Create the initial states (unit state for complex and substrate, singlet for H2)
rho_c = unit_state(spin_system_c)  # Initial state for the complex
rho_s = unit_state(spin_system_s)  # Initial state for the substrate
rho_H2 = singlet(spin_system_H2, 0, 1)  # Singlet state for H2

# Create an array for storing the magnetizations during evolution
magnetizations = np.empty((N_steps, isotopes_s.size), dtype=complex)  # Magnetization storage

# Evolve the system for the specified number of steps
for step in range(N_steps):

    # Dissociation of substrate
    rho_H2_old, rho_s_old = dissociate(spin_system_H2, spin_system_s, spin_system_c, rho_c, spin_map_H2, spin_map_s)

    # Calculate the complex where only the substrate is exchanged
    rho_c_new_s_old_H2 = associate(spin_system_H2, spin_system_s, spin_system_c, rho_H2_old, rho_s, spin_map_H2, spin_map_s)

    # Calculate the complex where both the substrate and H2 are exchanged
    rho_c_new_s_new_H2 = associate(spin_system_H2, spin_system_s, spin_system_c, rho_H2, rho_s, spin_map_H2, spin_map_s)

    # Account for the symmetry of pyridine
    # rho_s_old = (rho_s_old + permute_spins(spin_system_s, rho_s_old, rot_map)) / 2
    rho_s_old = (rho_s_old + permute_spins(spin_system_s, rho_s_old, perm_map)) / 2 # NOTE: Perttu's edit

    # Exchange process for free substrate
    rho_s = rho_s + c_c / c_s * dt * k_s * (rho_s_old - rho_s)

    # Exchange process for the complex
    rho_c = rho_c + dt * (
        (k_s - k_H2) * (rho_c_new_s_old_H2 - rho_c) +
        k_H2 * (rho_c_new_s_new_H2 - rho_c)
    )

    # Propagate the system forward in time
    rho_c = P_c @ rho_c
    rho_s = P_s @ rho_s

    # Measure the magnetization for each spin
    for i in range(isotopes_s.size):
        magnetizations[step, i] = measure(spin_system_s, rho_s, 'I_z', i)

# Plot the magnetizations and display the result
for i in range(isotopes_s.size):
    plt.plot(np.real(magnetizations[:, i]), label=f"Spin {i+1}, {isotopes_s[i]}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE Hyperpolarization of Pyridine")
plt.show()
plt.clf()
