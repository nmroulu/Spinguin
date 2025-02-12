"""
This is a more advanced example of a SABRE simulation of pyridine.
The spin system consists of the hydride protons and one pyridine ligand.
Chemical exchange and relaxation are taken into account.

Takes about 72 hours to run on a laptop with 11th generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from hyppy.spin_system import SpinSystem
from hyppy.hamiltonian import hamiltonian
from hyppy.relaxation import relaxation
from hyppy.propagation import propagator
from hyppy.states import unit_state, singlet, measure
from hyppy.chem import associate, dissociate, rotate_molecule
from hyppy.basis import ZQ_basis, ZQ_filter
from tqdm.auto import tqdm

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
isotopes_c = \
    np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N', '14N'])
isotopes_c1 = \
    np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_s = \
    np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
isotopes_H2 = \
    np.array(['1H', '1H'])

# Make spin maps for exchange processes
spin_map_c = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
spin_map_H2_and_s2 = (0, 1, 3, 5, 7, 9, 11, 13)
spin_map_s1 = (2, 4, 6, 8, 10, 12)
spin_map_H2_and_s1 = (0, 1, 2, 4, 6, 8, 10, 12)
spin_map_s2 = (3, 5, 7, 9, 11, 13)
spin_map_s = (2, 3, 4, 5, 6, 7)
spin_map_pH2 = (0, 1)

# Make a rotation map for pyridine that takes into account the symmetry
rot_map = (1, 0, 3, 2, 4, 5)

# Assign chemical shifts
chemical_shifts_c = np.array([-22.7, -22.7, 8.34, 8.34, 8.34, 8.34, 7.12, 7.12, 7.12, 7.12, 7.77, 7.77, 43.60, 38.43])
chemical_shifts_s = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
chemical_shifts_pH2 = np.array([4.2, 4.2])

# Assign scalar couplings
scalar_couplings_c = np.array([\
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
scalar_couplings_s = np.array([\
    [ 0,     0,      0,      0,      0,      0],
    [-1.04,  0,      0,      0,      0,      0],
    [ 4.85,  1.05,   0,      0,      0,      0],
    [ 1.05,  4.85,   0.71,   0,      0,      0],
    [ 1.24,  1.24,   7.55,   7.55,   0,      0],
    [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
])

# Assign coordinates
test_dir = os.path.dirname(__file__)
xyz_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine.xyz')
xyz_s = os.path.join(test_dir, 'example_data', 'pyridine.xyz')
sh_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine_shielding.txt')
sh_s = os.path.join(test_dir, 'example_data', 'pyridine_shielding.txt')
efg_c = os.path.join(test_dir, 'example_data', 'sabre_pyridine_efg.txt')
efg_s = os.path.join(test_dir, 'example_data', 'pyridine_efg.txt')

# Initialize the spin systems
spin_system_c = SpinSystem(isotopes_c, chemical_shifts_c, scalar_couplings_c, xyz_c, sh_c, efg_c, max_spin_order)
spin_system_c1 = SpinSystem(isotopes_c1, max_spin_order=max_spin_order)
spin_system_s = SpinSystem(isotopes_s, chemical_shifts_s, scalar_couplings_s, xyz_s, sh_s, efg_s, max_spin_order)
spin_system_H2 = SpinSystem(isotopes_H2)

# Make the Hamiltonians
H_c = hamiltonian(spin_system_c, magnetic_field)
H_s = hamiltonian(spin_system_s, magnetic_field)

# Make the Relaxation superoperators
R_c = relaxation(spin_system_c, H_c, magnetic_field, tau_c_c, include_sr2k=True)
R_s = relaxation(spin_system_s, H_s, magnetic_field, tau_c_s, include_sr2k=True)

# Switch to ZQ basis
ZQ_basis(spin_system_c)
ZQ_basis(spin_system_c1)
ZQ_basis(spin_system_s)
ZQ_basis(spin_system_H2)

# Change the Hamiltonians and the Relaxation superoperators to the ZQ basis
H_c = ZQ_filter(spin_system_c, H_c)
H_s = ZQ_filter(spin_system_s, H_s)
R_c = ZQ_filter(spin_system_c, R_c)
R_s = ZQ_filter(spin_system_s, R_s)

# Make the time propagator
P_c = propagator(time_step, H_c, R_c, custom_dot=True)
P_s = propagator(time_step, H_s, R_s, custom_dot=True)

# Create the initial state (unit state for complex and substrate, singlet for H2)
rho_c = unit_state(spin_system_c)
rho_s = unit_state(spin_system_s)
rho_H2 = singlet(spin_system_H2, 0, 1)

# Create an array for storing the magnetizations during evolution
magnetizations = np.empty((nsteps, isotopes_s.size), dtype=complex)

# Evolve the spin system for the number of steps
for step in tqdm(range(nsteps), miniters=nsteps/100, desc='Propagating...'):

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

    # Calculate complex where substrate 1 and pH2 is exchanged
    rho_c_new_s1_pH2 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_pH2_and_s, rho_s2, spin_map_H2_and_s1, spin_map_s2)

    # Calculate complex where substrate 2 and pH2 is exchanged
    rho_c_new_s2_pH2 = associate(spin_system_c1, spin_system_s, spin_system_c, rho_pH2_and_s, rho_s1, spin_map_H2_and_s2, spin_map_s1)

    # Take into account the symmetry of pyridine
    rho_s12 = (rho_s1 + rho_s2)/2
    rho_s12 = (rho_s12 + rotate_molecule(spin_system_s, rho_s12, rot_map))/2
    
    # Exchange process of free substrate
    rho_s = rho_s + c_c/c_s * time_step * k_s * (rho_s12 - rho_s)

    # Exchange process of complex
    rho_c = rho_c + time_step * (\
        (2*k_s - k_H2) * ((rho_c_new_s1 + rho_c_new_s2)/2 - rho_c) + \
        k_H2 * ((rho_c_new_s1_pH2 + rho_c_new_s2_pH2)/2 - rho_c))

    # Propagate the spin system
    rho_c = P_c @ rho_c
    rho_s = P_s @ rho_s

    # Calculate magnetization for substrate
    for i in range(spin_system_s.size):
        magnetizations[step][i] = measure(spin_system_s, rho_s, 'I_z', i)

# Plot the magnetizations and show the result
for i in range(isotopes_s.size):
    plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}, {isotopes_s[i]}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE-hyperpolarization of Pyridine")
plt.show()
plt.clf()
