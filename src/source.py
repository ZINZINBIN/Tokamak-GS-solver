from ._computes import *
            
# External Source : PF coils
class PFCoils(object):
    def __init__(self, 
                 Ic : float, 
                 Rc : float, 
                 Zc : float,
                 R_min : float,
                 R_max : float,
                 Z_min : float,
                 Z_max : float,
                 Nr : int,
                 Nz : int,
                 mu : float = MU):
        self.Ic = Ic
        self.Rc = Rc
        self.Zc = Zc
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.Nr = Nr
        self.Nz = Nz
        self.mu = mu
        
    def get_psi_boundary_from_coil(self, is_scaled = True):
        
        psi_boundary = np.zeros((self.Nr, self.Nz))
        
        dr = (self.R_max - self.R_min) / (self.Nr - 1)
        dz = (self.Z_max - self.Z_min) / (self.Nz - 1)
        
        for idx_r in range(0, self.Nr):
            for idx_z in range(0, self.Nz):
                R = self.R_min + idx_r * dr
                Z = self.Z_min + idx_z * dz
                psi_boundary[idx_r, idx_z] = GreenFunctionScaled(self.Rc, self.Zc, R, Z)
        
        if not is_scaled:
            psi_boundary *= self.Ic * self.mu
        
        return psi_boundary
