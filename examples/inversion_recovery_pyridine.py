# Imports
import numpy as np
import matplotlib.pyplot as plt
from spinguin.spin_system import SpinSystem
from spinguin.hamiltonian import hamiltonian
from spinguin.states import thermal_equilibrium, measure
from spinguin.relaxation import relaxation, thermalize
from spinguin.propagation import propagator, pulse
from spinguin.basis import ZQ_basis, ZQ_filter

# Example system
isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
chemical_shifts = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
scalar_couplings = np.array([\
    [ 0,     0,      0,      0,      0,      0],
    [-1.04,  0,      0,      0,      0,      0],
    [ 4.85,  1.05,   0,      0,      0,      0],
    [ 1.05,  4.85,   0.71,   0,      0,      0],
    [ 1.24,  1.24,   7.55,   7.55,   0,      0],
    [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
])
xyz = np.array([\
    [ 2.0495335, 0.0000000, -1.4916842],
    [-2.0495335, 0.0000000, -1.4916842],
    [ 2.1458878, 0.0000000,  0.9846086],
    [-2.1458878, 0.0000000,  0.9846086],
    [ 0.0000000, 0.0000000,  2.2681296],
    [ 0.0000000, 0.0000000, -1.5987077]
])
shielding = np.zeros((6, 3, 3))
shielding[5] = np.array([\
    [-406.20, 0.00,   0.00],
    [ 0.00,   299.44, 0.00],
    [ 0.00,   0.00,  -181.07]
])
efg = np.zeros((6, 3, 3))
efg[5] = np.array([\
    [0.3069, 0.0000,  0.0000],
    [0.0000, 0.7969,  0.0000],
    [0.0000, 0.0000, -1.1037]
])
spin_system = SpinSystem(isotopes, chemical_shifts, scalar_couplings, xyz, shielding, efg, max_spin_order=3)
field = 1
tau_c = 50e-12
temp = 273
time_step = 2e-3
nsteps = 50000

# Get the Hamiltonian
H = hamiltonian(spin_system, field)
R = relaxation(spin_system, H, field, tau_c, include_sr2k=True)

# Create the thermal equilibrium
rho = thermal_equilibrium(spin_system, temp, field)

# Apply 180-degree pulse for protons
pul_180 = pulse(spin_system, 'I_x', [0, 1, 2, 3, 4], angle=180)
rho = pul_180 @ rho

# Switch to ZQ subspace
ZQ_basis(spin_system)
H = ZQ_filter(spin_system, H)
R = ZQ_filter(spin_system, R)
rho = ZQ_filter(spin_system, rho)

# Thermalize R
R = thermalize(spin_system, R, field, temp)

# Get the propagator
P = propagator(time_step, H, R)

# Store the magnetizations to an array
magnetizations = np.zeros((nsteps, isotopes.size), dtype=complex)

# Simulation
for step in range(nsteps):

    # Propagate
    rho = P @ rho

    # Detect
    for idx in range(isotopes.size):
        magnetizations[step, idx] = measure(spin_system, rho, 'I_z', idx)

# Plot the magnetizations and show the result
for i in range(isotopes.size):
    plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("Inversion-recovery of Pyridine")
plt.show()
plt.clf()