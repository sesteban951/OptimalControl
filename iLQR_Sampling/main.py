##
#
# iLQR Sampling for pendulum swing-up
#
##

import torch 
import math
from cost import *
from pendulum import *
from cem import *


##################################################
# iLQR FUNCTIONS
##################################################



##################################################
# MAIN
##################################################

if __name__ == "__main__":

    # initilize the parameters
    params = {
        "m": 1.0,      # mass
        "l": 1.0,      # length
        "b": 0.1,      # damping
        "g": 9.81,     # gravity
        "dt": 0.02,    # time step
        "umax": 3.0,   # max torque
        "K": 256,      # iLQR iterations
        "eps": 1e-3,   # gradient mean sampling
    }

    # number of integration steps
    T = 250

    # initial state (downward at rest)
    x0 = torch.tensor([0.0, 0.0])

    # initialize CEM parameters
    cem_params = CEMParams(
        N=T,
        dt=params["dt"],
        B=256,
        n_elite=32,
        mu0=0.0,
        sigma0=1.0,
        sigma_min=0.01,
        sigma_max=1.0,
        iter=100
    )
    cem = CEM(cem_params)

    




