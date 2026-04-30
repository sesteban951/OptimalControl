##
#
# Pendulum Dynamics
#
##

import numpy as np

def f_cont(x, u):

    # dynamics
    theta = x[0]
    theta_dot = x[1]

    xdot = np.arr