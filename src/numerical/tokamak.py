import numpy as np
import scipy as sp
from src.numerical.compute import *
from src.numerical.grad_shafranov import *
from src.numerical.boundary import *
from src.numerical.profiles import *
from src.numerical.source import *

# Tokamak Environment
class Tokamak(object):
    def __init__(self, **kwargs):
        self.properties = PlasmaProperties(
            kwargs['Ip'],
            kwargs['beta_p'],
            kwargs['mu'],
            kwargs['lamda_0'],
            kwargs['beta_0'],
            kwargs['m'],
            kwargs['n'],
            kwargs['R_min'],
            kwargs['R_max'],
            kwargs['Z_min'],
            kwargs['Z_max'],
            kwargs['B0'],
            kwargs['R0'],
            kwargs['a'],
            kwargs['ad'],
            kwargs['kappa'],
            kwargs['q0'],
            kwargs['Nr'],
            kwargs['Nz']
        )
        
        # self.source = PFCoils(
        #     kwargs['Ic'],
        #     kwargs['Rc'],
        #     kwargs['Zc'],
        #     kwargs['R_min'],
        #     kwargs['R_max'],
        #     kwargs['Z_min'],
        #     kwargs['Z_max'],
        #     kwargs['Nr'],
        #     kwargs['Nz'],
        #     kwargs['mu']
        # )
        
        self.source_U = PFCoils(
            kwargs['Ic_U'],
            kwargs['Rc_U'],
            kwargs['Zc_U'],
            kwargs['R_min'],
            kwargs['R_max'],
            kwargs['Z_min'],
            kwargs['Z_max'],
            kwargs['Nr'],
            kwargs['Nz'],
            kwargs['mu']
        )
        
        self.source_L = PFCoils(
            kwargs['Ic_L'],
            kwargs['Rc_L'],
            kwargs['Zc_L'],
            kwargs['R_min'],
            kwargs['R_max'],
            kwargs['Z_min'],
            kwargs['Z_max'],
            kwargs['Nr'],
            kwargs['Nz'],
            kwargs['mu']
        )
        
        self.source = {
            "PFcoils_upper" : self.source_U,
            "PFcoils_lower" : self.source_L
        }
        
        self.args = kwargs
        
        solver = GSsolver(
            properties=self.properties,
            source = self.source,
            R_min = kwargs["R_min"],
            R_max = kwargs["R_max"],
            Z_min = kwargs["Z_min"],
            Z_max = kwargs["Z_max"],
            Nr = kwargs["Nr"],
            Nz = kwargs["Nz"],
            convergence=kwargs["convergence"]
        )
        
        self.GSsolver = solver
        self.psi = np.zeros((self.args['Nr'], self.args['Nz']))
        self.J_phi = np.zeros((self.args['Nr'], self.args['Nz']))
        
    def solve(self):
        self.GSsolver.solve(self.args["w"], self.args["iteration"])
        self.psi = self.GSsolver.psi
        self.J_phi = self.GSsolver.J_phi
        
    def plot_result(self):
        self.GSsolver.plotGrid(self.args["shell_number"], self.args["save_dir"])