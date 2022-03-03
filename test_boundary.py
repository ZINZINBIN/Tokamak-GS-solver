import argparse
import os
import sys
import math
from src.grad_shafranov import *
from src.tokamak import *

parser = argparse.ArgumentParser(description='Input argument for boundary plasma in Tokamak')

# Argument for Estimate J_phi
parser.add_argument('--beta_0', help = 'beta_0 for estimate J_phi, initial', type = float, default = 0.5)
parser.add_argument('--lamda_0',  help = 'lamda_0 for estimate J_phi, initial', type = float, default = 1.0)
parser.add_argument('--m',  help = 'm for estimate J_phi', type = int, default = 3)
parser.add_argument('--n',  help = 'n for estimate J_phi', type = int, default = 2)

# Argument for geometric properties of Tokamak
parser.add_argument('--R_min', help = 'Tokamak Min Radius', type = float, default = 1.0)
parser.add_argument('--R_max', help = 'Tokamak Max Radius', type = float, default = 2.0)
parser.add_argument('--Z_min', help = 'Tokamak Min Height', type = float, default = -0.25)
parser.add_argument('--Z_max', help = 'Tokamak Max Height', type = float, default = 0.25)
parser.add_argument('--a', help = 'a : Boundary condition', type = float, default = 0.3)
parser.add_argument('--ad', help = 'ad : Boundary condition', type = float, default = 0.1)
parser.add_argument('--kappa', help = 'kappa : Boundary condition', type = float, default = 2.2)
parser.add_argument('--q0', help = 'q0 : Boundary condition', type = float, default = 1.2)
parser.add_argument('--R0', help = 'R0 : Major Radius', type = float, default = 1.5)

# Argument for External Plasma Source
parser.add_argument('--Rc_U', help = 'PF coils Radius', type = float, default = 1.5)
parser.add_argument('--Zc_U', help = 'PF coils Height', type = float, default = 0.3)
parser.add_argument('--Ic_U', help = 'PF coils current(unit : A)', type = float, default = 1.0 * 10 **6)

parser.add_argument('--Rc_L', help = 'PF coils Radius', type = float, default = 1.5)
parser.add_argument('--Zc_L', help = 'PF coils Height', type = float, default = -0.3)
parser.add_argument('--Ic_L', help = 'PF coils current(unit : A)', type = float, default = 1.0 * 10 **6)

# Argument for Grad-Shafranov Equation
parser.add_argument('--Nr',  help = '2D Grid Interval Number of Radius Direction', type = int, default = 64)
parser.add_argument('--Nz',  help = '2D Grid Interval Number of Z-axis Direction', type = int, default = 64)
parser.add_argument('--w',  help = 'Relaxation Factor', type = float, default = 1.3)
parser.add_argument('--iteration',  help = 'Number of Iteration', type = int, default = 8)
parser.add_argument('--convergence',  help = 'Criteria for Convergence', type = int, default = 1e-6)

# Argument for plasma control parameter
parser.add_argument('--mu', help = 'Permeability', type = float, default = 4 * math.pi * 10 ** (-7) )
parser.add_argument('--Ip', help = 'Plasma Current(Toroidal, unit : A)', type = float, default = 1 * 10**6)
parser.add_argument('--beta_p', help = 'Beta_p', type = float, default = 2.0)
parser.add_argument('--B0', help = 'Magnetic Field(Toroidal, unit : T', type = float, default = 3.0)

# Argument for plotting psi and J_phi Graph
parser.add_argument('--shell_number', help = 'Contour Boundary Number', type = int, default = 8)
parser.add_argument('--save_dir',  help = 'Save Directory', type = str, default = './results/Boundary-Equilibrium.png')

kargs = vars(parser.parse_args())

if __name__ == "__main__":
    
    tokamak = Tokamak(**kargs)
    tokamak.solve()
    tokamak.plot_result()