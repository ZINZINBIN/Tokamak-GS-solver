import argparse
import os
import sys
import math
from src.grad_shafranov import *

parser = argparse.ArgumentParser(description='Input Argument For Free Boundary Tokamak Equilibrium')

# Argument for Estimate J_phi
parser.add_argument('--beta_0', help = 'beta_0 for estimate J_phi', type = float, default = 1.0)
parser.add_argument('--lamda_0',  help = 'lamda_0 for estimate J_phi', type = float, default = 1.0)
parser.add_argument('--m',  help = 'm for estimate J_phi', type = int, default = 1)
parser.add_argument('--n',  help = 'n for estimate J_phi', type = int, default = 1)

# Argument for geometric properties of Tokamak
parser.add_argument('--Rmin', help = 'Tokamak Min Radius', type = float, default = 1.0)
parser.add_argument('--Rmax', help = 'Tokamak Max Radius', type = float, default = 2.0)
parser.add_argument('--Zmin', help = 'Tokamak Min Height', type = float, default = 0.0)
parser.add_argument('--Zmax', help = 'Tokamak Max Height', type = float, default = 0.5)

# Argument for External Plasma Source
parser.add_argument('--Rc', help = 'PF coils Radius', type = float, default = 1.5)
parser.add_argument('--Zc', help = 'PF coils Height', type = float, default = 0.25)
parser.add_argument('--Ic', help = 'PF coils current', type = float, default = 1.0)

# Argument for Grad-Shafranov Equation
parser.add_argument('--Nr',  help = '2D Grid Interval Number of Radius Direction', type = int, default = 128)
parser.add_argument('--Nz',  help = '2D Grid Interval Number of Z-axis Direction', type = int, default = 128)
parser.add_argument('--w',  help = 'Relaxation Factor', type = float, default = 1.3)
parser.add_argument('--iteration',  help = 'Number of Iteration', type = int, default = 32)
parser.add_argument('--convergence',  help = 'Criteria for Convergence', type = int, default = 1e-7)

# Argument for Other properties
parser.add_argument('--mu', help = 'Permeability', type = float, default = 4 * math.pi * 10 ** (-7) )

# Argument for plotting psi and J_phi Graph
parser.add_argument('--shell_number', help = 'Contour Boundary Number', type = int, default = 16)
parser.add_argument('--save_dir',  help = 'Save Directory', type = str, default = './results/Free-Boundary-Equilibrium.png')

args = parser.parse_args()

if __name__ == "__main__":

    PF_source = {
        'PFcoil_current' : args.Ic,
        'rc':args.Rc,
        'zc':args.Zc
    }
    
    gsSolver = GSsolverFreeBoundary(
        args.Rmin,
        args.Rmax,
        args.Zmin,
        args.Zmax,
        args.m,
        args.n,
        args.beta_0,
        args.lamda_0,
        args.mu,
        args.w,
        PF_source
    )
    
    gsSolver.solve(args.Nr, args.Nz, args.iteration, args.convergence)
    
    gsSolver.plotGrid(args.shell_number, args.save_dir)