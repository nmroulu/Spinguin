"""
This is a more advanced example of a SABRE simulation of pyridine.
The spin system consists of the hydride protons and one pyridine ligand.
Chemical exchange and relaxation are taken into account.

Takes about 10-15 minutes to run on a laptop with 11th generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from spinguin import SpinSystem, hamiltonian, relaxation, propagator, unit_state, singlet, measure, associate, dissociate, rotate_molecule, ZQ_basis, ZQ_filter

# Simulation settings
max_spin_order = 4
magnetic_field = 7e-3
time_step = 2e-3
nsteps = 30000
k_H2 = 1.6
k_s = 10
tau_c_c = 43e-12
tau_c_s = 5.7e-12
c_c = 0.0005
c_s = 0.015

# Assign isotopes
isotopes_c = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_s = np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_H2 = np.array(['1H', '1H'])

# Make spin maps for exchange processes
spin_map_c = (0, 1, 2, 3, 4, 5, 6, 7)
spin_map_s = (2, 3, 4, 5, 6, 7)
spin_map_H2 = (0, 1)

# Make a rotation map for pyridine that takes into account the symmetry
rot_map = (1, 0, 3, 2, 4, 5)

# Assign chemical shifts
chemical_shifts_c = np.array([-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77, 43.60])
chemical_shifts_s = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
chemical_shifts_H2 = np.array([4.2, 4.2])

# Assign scalar couplings
scalar_couplings_c = np.array([\
    [ 0,     0,      0,      0,      0,      0,      0,     0],
    [-6.53,  0,      0,      0,      0,      0,      0,     0],
    [ 0.00,  1.66,   0,      0,      0,      0,      0,     0],
    [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0],					
    [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0,     0],					
    [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0,     0],			
    [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0,     0],
    [-0.30,  15.91,  4.47,   0.04,   1.79,   0,     -0.46,  0]
])
scalar_couplings_s = np.array([\
    [ 0,     0,      0,      0,      0,      0],
    [-1.04,  0,      0,      0,      0,      0],
    [ 4.85,  1.05,   0,      0,      0,      0],
    [ 1.05,  4.85,   0.71,   0,      0,      0],
    [ 1.24,  1.24,   7.55,   7.55,   0,      0],
    [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
])

# Assign coordinates
xyz_c = np.array([\
    [ 0.9649170,  1.2271534, -1.2031835],
    [ 1.9547078, -0.4342818, -0.1922623],
    [-2.5492743, -0.9988969, -0.2721286],
    [-1.3895773, -2.6746310, -1.2331514],
    [-4.5704762, -1.0068808, -1.6740361],
    [-1.5724294, -5.0099155, -0.4837575],
    [-4.5894708,  0.3577455, -3.7835019],
    [-1.4702745,  0.2327446, -1.5095832]
])
xyz_s = np.array([\
    [ 2.0495335, 0.0000000, -1.4916842],
    [-2.0495335, 0.0000000, -1.4916842],
    [ 2.1458878, 0.0000000,  0.9846086],
    [-2.1458878, 0.0000000,  0.9846086],
    [ 0.0000000, 0.0000000,  2.2681296],
    [ 0.0000000, 0.0000000, -1.5987077]
])

# Assign shielding tensors
shielding_c = np.zeros((8, 3, 3))
shielding_c[7] = np.array([\
    [-134.70, -123.93, -49.86],
    [-147.79,  64.47,   221.96],
    [-62.63,   223.57, -60.57]
])
shielding_s = np.zeros((6, 3, 3))
shielding_s[5] = np.array([\
    [-406.20, 0.00,   0.00],
    [ 0.00,   299.44, 0.00],
    [ 0.00,   0.00,  -181.07]
])

# Assign EFG tensors
efg_c = np.zeros((8, 3, 3))
efg_c[7] = np.array([\
    [-0.3426, -0.0417, -0.4514],
    [-0.0417,  0.3727,  0.1186],
    [-0.4514,  0.1186, -0.0301]
])
efg_s = np.zeros((6, 3, 3))
efg_s[5] = np.array([\
    [0.3069, 0.0000,  0.0000],
    [0.0000, 0.7969,  0.0000],
    [0.0000, 0.0000, -1.1037]
])

# Initialize the spin systems
spin_system_c = SpinSystem(isotopes_c, chemical_shifts_c, scalar_couplings_c, xyz_c, shielding_c, efg_c, max_spin_order)
spin_system_s = SpinSystem(isotopes_s, chemical_shifts_s, scalar_couplings_s, xyz_s, shielding_s, efg_s, max_spin_order)
spin_system_H2 = SpinSystem(isotopes_H2)

# Make the Hamiltonians
H_c = hamiltonian(spin_system_c, magnetic_field)
H_s = hamiltonian(spin_system_s, magnetic_field)

# Make the Relaxation superoperators
R_c = relaxation(spin_system_c, H_c, magnetic_field, tau_c_c, include_sr2k=True)
R_s = relaxation(spin_system_s, H_s, magnetic_field, tau_c_s, include_sr2k=True)

# Switch to ZQ basis
ZQ_map_c = ZQ_basis(spin_system_c)
ZQ_map_s = ZQ_basis(spin_system_s)
ZQ_basis(spin_system_H2)

# Change the Hamiltonians and the Relaxation superoperators to the ZQ basis
H_c = ZQ_filter(spin_system_c, H_c, ZQ_map_c)
H_s = ZQ_filter(spin_system_s, H_s, ZQ_map_s)
R_c = ZQ_filter(spin_system_c, R_c, ZQ_map_c)
R_s = ZQ_filter(spin_system_s, R_s, ZQ_map_s)

# Make the time propagator
P_c = propagator(time_step, H_c, R_c)
P_s = propagator(time_step, H_s, R_s)

# Create the initial state (unit state for complex and substrate, singlet for H2)
rho_c = unit_state(spin_system_c)
rho_s = unit_state(spin_system_s)
rho_H2 = singlet(spin_system_H2, 0, 1)

# Create an array for storing the magnetizations during evolution
magnetizations = np.empty((nsteps, isotopes_s.size), dtype=complex)

# Evolve for the number of steps
for step in range(nsteps):

    # Dissociation of substrate
    rho_H2_old, rho_s_old = dissociate(spin_system_H2, spin_system_s, spin_system_c, rho_c, spin_map_H2, spin_map_s)

    # Calculate complex where only substrate is exchanged
    rho_c_new_s_old_H2 = associate(spin_system_H2, spin_system_s, spin_system_c, rho_H2_old, rho_s, spin_map_H2, spin_map_s)

    # Calculate complex where both substrate and H2 is exchanged
    rho_c_new_s_new_H2 = associate(spin_system_H2, spin_system_s, spin_system_c, rho_H2, rho_s, spin_map_H2, spin_map_s)

    # Take into account the symmetry of pyridine
    rho_s_old = (rho_s_old + rotate_molecule(spin_system_s, rho_s_old, rot_map))/2

    # Exchange process of free substrate
    rho_s = rho_s + c_c/c_s * time_step * k_s * (rho_s_old - rho_s)

    # Exchange process of complex
    rho_c = rho_c + time_step * (\
        (k_s - k_H2) * (rho_c_new_s_old_H2 - rho_c) + \
        k_H2 * (rho_c_new_s_new_H2 - rho_c))

    # Propagate the system forward in time
    rho_c = P_c @ rho_c
    rho_s = P_s @ rho_s

    # Measure the magnetization for each spin
    for i in range(isotopes_s.size):
        magnetizations[step, i] = measure(spin_system_s, rho_s, 'I_z', i)

# Plot the magnetizations and show the result
for i in range(isotopes_s.size):
    plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}, {isotopes_s[i]}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE-hyperpolarization of Pyridine")
plt.show()
plt.clf()
