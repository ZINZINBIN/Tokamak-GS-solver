from ._computes import *

# Boundary Object
# D-shape
class Boundary(object):
    def __init__(self, 
                 R_min : float,
                 R_max : float,
                 Z_min : float,
                 Z_max : float,
                 B0 : float,
                 R0 : float,
                 a : float,
                 ad : float,
                 kappa : float,
                 q0 : float,
                 Nr : int,
                 Nz : int):
        self.kappa = kappa # elongation
        self.q0 = q0
        self.R0 = R0
        self.a = a
        self.ad = ad # triangularity
        self.Nr = Nr
        self.Nz = Nz
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.psi_b = None
        self.psi_a = None
        self.B0 = B0
        
        self.R_center = 0.5 * (R_min + R_max)
        self.Z_center = 0.5 * (Z_min + Z_max)
    
        self.grid = np.zeros((Nr,Nz))
        
    def _update_psi_b(self, psi):
        
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        psi_b = 0
        for idx_r in range(1,Nr-1):
            for idx_z in range(1,Nz-1):
                if self._check_x_point(psi, idx_r, idx_z):
                    psi_b = psi[idx_r, idx_z]
                    print("update psi_b : ", psi_b)
                    break
                    
        if psi_b == 0:
            s = self.psi_grad_abs.reshape(-1,).argsort()
            hessian = self.get_hessian_matrix(psi).reshape(-1,)[s]
            idx_satisfied = (hessian < 0).nonzero()
            psi_b = psi.reshape(-1,)[idx_satisfied[0].min()]
                    
        self.psi_b = psi_b
       
    def get_hessian_matrix(self, psi):
        hessian = np.zeros_like(psi) 
        
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        dR = (self.R_max - self.R_min) / (Nr -1)
        dZ = (self.Z_max - self.Z_min) / (Nz -1)
        
        for idx_r in range(1,Nr-1):
            for idx_z in range(1, Nz-1):
                
                dPsi_dR = psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z]
                dPsi_dR /= (2*dR)
                
                dPsi_dZ = psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1]
                dPsi_dZ /= (2*dZ)
                
                # S(R,Z) > 0 : magnetic axis
                # S(R,Z) < 0 : x-point
                dPsi_dR2 = psi[idx_r + 1, idx_z] + psi[idx_r - 1, idx_z] - 2 * psi[idx_r, idx_z] 
                dPsi_dR2 /= dR**2
                
                dPsi_dZ2 = psi[idx_r, idx_z + 1] + psi[idx_r, idx_z - 1] - 2 * psi[idx_r, idx_z] 
                dPsi_dZ2 /= dZ**2
                
                dPsi_dRdZ = psi[idx_r + 1, idx_z + 1] + psi[idx_r - 1, idx_z - 1] - \
                    psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z] - psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1] + \
                    2 * psi[idx_r, idx_z]
                    
                dPsi_dRdZ /= (2 * dR * dZ)
                S = dPsi_dR2 * dPsi_dZ2 - dPsi_dRdZ ** 2
                hessian[idx_r, idx_z] = S
        
        return hessian
        
    def _update_psi_grad(self, psi):
        psi_grad_r = compute_derivative_matrix(psi, self.R_min, self.R_max, self.Z_min, self.Z_max, 0)
        psi_grad_z = compute_derivative_matrix(psi, self.R_min, self.R_max, self.Z_min, self.Z_max, 1)
        self.psi_grad_abs = np.sqrt(psi_grad_r ** 2 + psi_grad_z ** 2)
        
    def _update_psi_a(self, psi):
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        psi_a = 0
        for idx_r in range(1,Nr-1):
            for idx_z in range(1,Nz-1):
                if self._check_axis(psi, idx_r, idx_z):
                    psi_a = psi[idx_r, idx_z]
                    print("update psi_a : ", psi_a)
                    break
                
        if psi_a == 0:
            s = self.psi_grad_abs.reshape(-1,).argsort()
            hessian = self.get_hessian_matrix(psi).reshape(-1,)[s]
            idx_satisfied = (hessian > 0).nonzero()
            psi_a = psi.reshape(-1,)[idx_satisfied[0].min()]
        
        self.psi_a = psi_a
        
    def _check_x_point(self, psi, idx_r, idx_z):
        # return whether (idx_r, idx_z) is x-point
        # Gradient Psi = 0 at (idx_r, idx_z) => field null 
        
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        dR = (self.R_max - self.R_min) / (Nr -1)
        dZ = (self.Z_max - self.Z_min) / (Nz -1)
        
        # check whether Gradient Psi = 0
        
        dPsi_dR = psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z]
        dPsi_dR /= (2*dR)
        
        dPsi_dZ = psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1]
        dPsi_dZ /= (2*dZ)
        
        dPsi_norm = np.sqrt(dPsi_dR ** 2 + dPsi_dZ ** 2)
        
        # S(R,Z) > 0 : magnetic axis
        # S(R,Z) < 0 : x-point
        dPsi_dR2 = psi[idx_r + 1, idx_z] + psi[idx_r - 1, idx_z] - 2 * psi[idx_r, idx_z] 
        dPsi_dR2 /= dR**2
        
        dPsi_dZ2 = psi[idx_r, idx_z + 1] + psi[idx_r, idx_z - 1] - 2 * psi[idx_r, idx_z] 
        dPsi_dZ2 /= dZ**2
        
        dPsi_dRdZ = psi[idx_r + 1, idx_z + 1] + psi[idx_r - 1, idx_z - 1] - \
            psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z] - psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1] + \
            4 * psi[idx_r, idx_z]
            
        dPsi_dRdZ /= (2 * dR * dZ)
        
        S = dPsi_dR2 * dPsi_dZ2 - dPsi_dRdZ ** 2
        condition = dPsi_norm < EPS_GRAD and S < 0
        
        return condition
    
    def _check_axis(self, psi, idx_r, idx_z):
        # return whether (idx_r, idx_z) is magnetic axis
        # Gradient Psi = 0 at (idx_r, idx_z) => field null 
        
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        dR = (self.R_max - self.R_min) / (Nr -1)
        dZ = (self.Z_max - self.Z_min) / (Nz -1)
        
        # check whether Gradient Psi = 0
        
        dPsi_dR = psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z]
        dPsi_dR /= (2*dR)
        
        dPsi_dZ = psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1]
        dPsi_dZ /= (2*dZ)
        
        dPsi_norm = np.sqrt(dPsi_dR ** 2 + dPsi_dZ ** 2)
        
        # S(R,Z) > 0 : magnetic axis
        # S(R,Z) < 0 : x-point
        dPsi_dR2 = psi[idx_r + 1, idx_z] + psi[idx_r - 1, idx_z] - 2 * psi[idx_r, idx_z] 
        dPsi_dR2 /= dR**2
        
        dPsi_dZ2 = psi[idx_r, idx_z + 1] + psi[idx_r, idx_z - 1] - 2 * psi[idx_r, idx_z] 
        dPsi_dZ2 /= dZ**2
        
        dPsi_dRdZ = psi[idx_r + 1, idx_z + 1] + psi[idx_r - 1, idx_z - 1] - \
            psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z] - psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1] + \
            4 * psi[idx_r, idx_z]
            
        dPsi_dRdZ /= (2 * dR * dZ)
        
        S = dPsi_dR2 * dPsi_dZ2 - dPsi_dRdZ ** 2
        
        condition = dPsi_norm < EPS_GRAD and S > 0
        
        return condition
                
    def _update_grid(self):
        # update plasma boundary
        
        Nr = self.Nr
        Nz = self.Nz
        
        dR = (self.R_max - self.R_min) / (Nr - 1)
        dZ = (self.Z_max - self.Z_min) / (Nz - 1)
        
        for idx_r in range(0,Nr):
            r = self.R_min + idx_r * dR
            
            if 2 * self.R0 ** 2 * self.kappa * self.q0 / self.B0 * self.psi_b - 0.25 * self.kappa ** 2 *(r**2 - self.R0**2) > 0:
                
                zp = math.sqrt(
                    2 * self.R0 ** 2 * self.kappa * self.q0 / self.B0 * self.psi_b - 
                    0.25 * self.kappa ** 2 *(r**2 - self.R0**2)
                )
                
                zp /= r
                zm = zp * (-1)
                
                zp += self.Z_center
                zm += self.Z_center
                
                zm_idx = 0
                zp_idx = Nz - 1
                
                for idx_z in range(0,Nz-1):
                    
                    zl = self.Z_min + idx_z * dZ
                    zr = self.Z_min + (idx_z + 1) * dZ
                
                    if zl <= zp and zr >=zp:
                        zp_idx = idx_z
                        # self.grid[idx_r, idx_z] = 1
                    elif zl <= zm and zm >=zp:
                        zm_idx = idx_z
                        # self.grid[idx_r, idx_z] = 1
                        
                self.grid[idx_r, zm_idx : zp_idx] = 1
                
            else:
                pass