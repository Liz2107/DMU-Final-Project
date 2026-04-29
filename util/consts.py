import numpy as np

R = 384400 # Earth-Moon distance [km]
T = 27.321582 * 86400 # Earth-Moon orbital period [s]
M_M = 7.3483E22 # Lunar mass [kg]
M_E = 5.9742E24 # Earth mass [kg]

mu = M_M / (M_E + M_M)
DU = R # CR3BP distance scale [km]
TU = T / (2 * np.pi) # CR3BP time scale [s]
VU = DU / TU # CR3BP velocity scale [km / s]
