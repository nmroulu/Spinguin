"""
This is a more advanced example of a SABRE simulation of pyridine.
The spin system consists of the hydride protons and one pyridine ligand.
Chemical exchange and relaxation are taken into account.

Note: This simulation may take approximately 72 hours to run on a laptop with an 11th generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from spinguin import SpinSystem, hamiltonian, relaxation, propagator, unit_state, singlet, measure, associate, dissociate, permute_spins, ZQ_basis_map, ZQ_filter
from tqdm.auto import tqdm

# Simulation settings
max_spin_order = 4  # Maximum spin order for the simulation
B0 = 7e-3  # Magnetic field strength in Tesla
dt = 2e-3  # Time step for propagation in seconds
N_steps = 30000  # Number of propagation steps
k_H2 = 1.6  # Rate constant for H2 exchange
k_s = 10  # Rate constant for substrate exchange
tau_c_c = 43e-12  # Correlation time for the complex in seconds
tau_c_s = 5.7e-12  # Correlation time for the substrate in seconds
c_c = 0.0005  # Concentration of the complex
c_s = 0.015  # Concentration of the substrate

# Assign isotopes
# Isotopes for the complex, substrate, and H2
isotopes_c = \
    np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N', '14N'])
isotopes_c1 = \
    np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_s = \
    np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_H2 = \
    np.array(['1H', '1H'])

# Create spin maps for exchange processes
# These maps define how spins are exchanged between different components
spin_map_c = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
spin_map_H2_and_s2 = (0, 1, 3, 5, 7, 9, 11, 13)
spin_map_s1 = (2, 4, 6, 8, 10, 12)
spin_map_H2_and_s1 = (0, 1, 2, 4, 6, 8, 10, 12)
spin_map_s2 = (3, 5, 7, 9, 11, 13)
spin_map_s = (2, 3, 4, 5, 6, 7)
spin_map_pH2 = (0, 1)

# Create a permutation map for pyridine that accounts for symmetry
perm_map = (0, 1, 2, 3, 4, 5)  # NOTE: Perttu's edit

# Assign chemical shifts
# Chemical shifts for the complex, substrate, and H2
chemical_shifts_c = np.array([-22.7, -22.7, 8.34, 8.34, 8.34, 8.34, 7.12, 7.12, 7.12, 7.12, 7.77, 7.77, 43.60, 38.43])
chemical_shifts_s = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
chemical_shifts_pH2 = np.array([4.2, 4.2])

# Assign scalar couplings
# Scalar coupling matrices for the complex and substrate
J_couplings_c = np.array([\
    [ 0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,      0,      0,      0],
    [-6.53,  0,      0,      0,      0,      0,      0,     0,      0,      0,      0,      0,      0,      0],
    [ 0.00,  1.66,   0,      0,      0,      0,      0,     0,      0,      0,      0,      0,      0,      0],
    [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0,      0,      0,      0,      0,      0,      0],
    [-0.06,  1.59,  -0.12,  -0.01,   0,      0,      0,     0,      0,      0,      0,      0,      0,      0],
    [ 1.49,	 0.35,	 0.04,	-0.12,	 0.01,   0,      0,     0,      0,      0,      0,      0,      0,      0],						
    [-0.09,	 0.35,	 6.03,	 0.14,	 0.94,	-0.05,	 0,     0,      0,      0,      0,      0,      0,      0],					
    [ 0.38, -0.13,	 0.09,	 5.93,	-0.05,	 0.94,	0.06,	0,      0,      0,      0,      0,      0,      0],			
    [-0.03,  0.35,	 0.93,	-0.03,	 6.00,	-0.05,	1.08,  -0.04,	0,      0,      0,      0,      0,      0],		
    [ 0.29, -0.10,	-0.05,	 0.92,	-0.05,	 6.09,  -0.04,  1.05,  -0.06,	0,      0,      0,      0,      0],	
    [ 0.01,	 0.03,	 1.12,	-0.02,	 1.11,	-0.06,	7.75,  -0.01,   7.77,  -0.06,	0,      0,      0,      0],
    [ 0.04,	 0.00,	-0.01,	 1.03,	-0.06,	 1.13,  -0.03,  7.82,  -0.06,	7.79,  -0.04,	0,      0,      0],	
    [-0.30,	 15.91,	 4.47,	 0.04,	 4.21,	 0.01,	1.79,   0,      1.87,   0,     -0.46,	0,      0,      0],
    [ 15.84,-0.31,	 0.03,	 4.46,	-0.05,	 4.34,  0,  	1.77,   0,      1.92,   0,     -0.48,	0,      0]
])
J_couplings_s = np.array([\
    [ 0,     0,      0,      0,      0,      0],
    [-1.04,  0,      0,      0,      0,      0],
    [ 4.85,  1.05,   0,      0,      0,      0],
    [ 1.05,  4.85,   0.71,   0,      0,      0],
    [ 1.24,  1.24,   7.55,   7.55,   0,      0],
    [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
])

# Assign coordinates
# Paths to coordinate and shielding data files
test_dir = os.path.dirname(__file__)
xyz_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine.xyz')
xyz_s = os.path.join(test_dir, 'example_data', 'pyridine.xyz')
sh_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine_shielding.txt')
sh_s = os.path.join(test_dir, 'example_data', 'pyridine_shielding.txt')
efg_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine_efg.txt')
efg_s = os.path.join(test_dir, 'example_data', 'pyridine_efg.txt')

# Initialize the spin systems
# Create spin systems for the complex, substrate, and H2
spin_system_c = SpinSystem(isotopes_c, chemical_shifts_c, J_couplings_c, xyz_c, sh_c, efg_c, max_spin_order)
spin_system_c1 = SpinSystem(isotopes_c1, max_spin_order=max_spin_order)
spin_system_s = SpinSystem(isotopes_s, chemical_shifts_s, J_couplings_s, xyz_s, sh_s, efg_s, max_spin_order)
spin_system_H2 = SpinSystem(isotopes_H2)

# Create the Hamiltonians
# Hamiltonians for the complex and substrate
H_c = hamiltonian(spin_system_c, B0)
H_s = hamiltonian(spin_system_s, B0)

# Create the Relaxation superoperators
# Relaxation superoperators for the complex and substrate
R_c = relaxation(spin_system_c, H_c, B0, tau_c_c, include_sr2k=True)
R_s = relaxation(spin_system_s, H_s, B0, tau_c_s, include_sr2k=True)

# Switch to ZQ basis
# Transform spin systems to the zero-quantum (ZQ) basis
ZQ_map_c = ZQ_basis_map(spin_system_c)
ZQ_basis_map(spin_system_c1)
ZQ_map_s = ZQ_basis_map(spin_system_s)
ZQ_basis_map(spin_system_H2)

# Transform the Hamiltonians and Relaxation superoperators to the ZQ basis
H_c = ZQ_filter(spin_system_c, H_c, ZQ_map_c)
H_s = ZQ_filter(spin_system_s, H_s, ZQ_map_s)
R_c = ZQ_filter(spin_system_c, R_c, ZQ_map_c)
R_s = ZQ_filter(spin_system_s, R_s, ZQ_map_s)

# Create the time propagator
# Propagators for the complex and substrate
P_c = propagator(dt, H_c, R_c, custom_dot=True)
P_s = propagator(dt, H_s, R_s, custom_dot=True)

# Create the initial state (unit state for complex and substrate, singlet for H2)
rho_c = unit_state(spin_system_c)
rho_s = unit_state(spin_system_s)
rho_H2 = singlet(spin_system_H2, 0, 1)

# Create an array for storing the magnetizations during evolution
magnetizations = np.zeros((N_steps, isotopes_s.size), dtype=complex)

# Evolve the spin system for the specified number of steps
for step in tqdm(range(N_steps), miniters=N_steps/100, desc='Propagating...'):

    # Dissociation of substrate 1
    rho_H2_and_s2, rho_s1 = dissociate(spin_system_c1, spin_system_s, spin_system_c, rho_c, spin_map_H2_and_s2, spin_map_s1)

    # Dissociation of substrate 2
    rho_H2_and_s1, rho_s2 = dissociate(spin_system_c1, spin_system_s, spin_system_c, rho_c, spin_map_H2_and_s1, spin_map_s2)

    # Calculate complex where substrate 1 is exchanged
    rho_c_new_s1 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_H2_and_s2, rho_s, spin_map_H2_and_s2, spin_map_s1)

    # Calculate complex where substrate 2 is exchanged
    rho_c_new_s2 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_H2_and_s1, rho_s, spin_map_H2_and_s1, spin_map_s2)

    # Calculate complex with exchanged substrate and pH2
    rho_pH2_and_s = associate(spin_system_H2, spin_system_s, spin_system_c1, rho_H2, rho_s, spin_map_pH2, spin_map_s)

    # Calculate complex where substrate 1 and pH2 are exchanged
    rho_c_new_s1_pH2 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_pH2_and_s, rho_s2, spin_map_H2_and_s1, spin_map_s2)

    # Calculate complex where substrate 2 and pH2 are exchanged
    rho_c_new_s2_pH2 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_pH2_and_s, rho_s1, spin_map_H2_and_s2, spin_map_s1)

    # Account for the symmetry of pyridine
    rho_s12 = (rho_s1 + rho_s2)/2
    # rho_s12 = (rho_s12 + permute_spins(spin_system_s, rho_s12, rot_map))/2
    rho_s12 = permute_spins(spin_system_s, rho_s12, perm_map) # NOTE: Perttu's edit
    
    # Exchange process of free substrate
    rho_s = rho_s + c_c/c_s * dt * k_s * (rho_s12 - rho_s)

    # Exchange process of complex
    rho_c = rho_c + dt * (\
        (2*k_s - k_H2) * ((rho_c_new_s1 + rho_c_new_s2)/2 - rho_c) + \
        k_H2 * ((rho_c_new_s1_pH2 + rho_c_new_s2_pH2)/2 - rho_c))

    # Propagate the spin system
    rho_c = P_c @ rho_c
    rho_s = P_s @ rho_s

    # Calculate magnetization for substrate
    for i in range(spin_system_s.size):
        magnetizations[step][i] = measure(spin_system_s, rho_s, 'I_z', i)

# Plot the magnetizations and display the result
for i in range(isotopes_s.size):
    plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}, {isotopes_s[i]}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE-hyperpolarization of Pyridine")
plt.show()
plt.clf()
