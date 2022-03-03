import numpy as np

from .boundary import *
from ._computes import *

# Properties : Plasma Inside Tokamak
class PlasmaProperties(Boundary):
    def __init__(self, 
                 Ip : float, 
                 beta_p : float, 
                 mu : float, 
                 lamda : float,
                 beta_0 : float,
                 m : int,
                 n : int,
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
                 Nz : int
                 ):
        
        super().__init__(R_min, R_max, Z_min, Z_max, B0, R0, a, ad, kappa, q0, Nr, Nz)
        self.mu = mu
        self.Ip = Ip
        self.beta_p = beta_p
        self.lamda = lamda
        self.beta_0 = beta_0
        self.m = m
        self.n = n
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.B0 = B0
        self.a = a
        self.ad = ad
        self.kappa = kappa
        self.q0 = q0
        self.Nr = Nr
        self.Nz = Nz
        
    def update_plasma_boundary(self, psi):
        # update Boundary grid after updating psi_a and psi_b
        self._update_psi_grad(psi)
        self._update_psi_a(psi)
        self._update_psi_b(psi)
        self._update_grid()
        
    def set_psi(self, psi):
        self.psi = psi
        
    def get_J_phi_plasma(self, psi, is_scaled = True):
        Nr = psi.shape[0]
        Nz = psi.shape[1]
        
        dR = (self.R_max - self.R_min) / (Nr - 1)
        dZ = (self.Z_max - self.Z_min) / (Nz - 1)
        
        J_phi_plasma = np.zeros_like(psi)
        
        for idx_r in range(0, Nr):
            for idx_z in range(0, Nz): 
                R = self.R_min + dR * idx_r
                Z = self.Z_min + dZ * idx_z
                
                J_phi_plasma[idx_r, idx_z] = compute_J_phi_plasma(
                    psi[idx_r, idx_z],
                    self.psi_a,
                    self.psi_b,
                    R,
                    self.R0,
                    None,
                    self.beta_0,
                    self.m,
                    self.n
                )
                
        J_phi_plasma = np.multiply(J_phi_plasma, self.grid)
        
        if not is_scaled:
            J_phi_plasma *= self.lamda
                
        # lamda : J_phi scale factor를 제외한 값을 반환
        return J_phi_plasma
    
    def get_psi_boundary_from_plasma(self, psi, is_scaled = True):  
        
        psi_boundary = np.zeros((self.Nr, self.Nz))
        J_phi = self.get_J_phi_plasma(psi, is_scaled=is_scaled)
        
        # if(np.allclose(J_phi, np.zeros_like(J_phi), atol = 1e-6)):
        #     print("profiles error : get_J_phi_plasma has zero value!")
        
        dR = (self.R_max - self.R_min) / (self.Nr - 1)
        dZ = (self.Z_max - self.Z_min) / (self.Nz - 1)
        
        for idx_r in range(0, self.Nr):
            for idx_z in range(0, self.Nz):
                R = self.R_min + dR * idx_r # Rb
                Z = self.Z_min + dZ * idx_z # Zb
                G = GreenFunctionMatrixScaled(R, Z, self.R_min, self.Z_min, self.R_max, self.Z_max, self.Nr, self.Nz)
                
                A = np.multiply(J_phi, G)
                psi_boundary[idx_r, idx_z] = Compute2DIntegral(A, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        if not is_scaled:
            psi_boundary *= self.mu 
        
        return psi_boundary
        
    def update_beta_0(self, psi, is_scaled = True):
        
        psi_b = np.ones_like(psi) * self.psi_b
        psi_b -= psi
        psi_b = np.multiply(psi_b, self.grid)
        psi_integral = Compute2DIntegral(psi_b, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        R_matrix = np.zeros_like(psi)
        R_inv_matrix = np.zeros_like(psi)
        dR = (self.R_max - self.R_min) / (self.Nr - 1)
        
        for idx_r in range(0, self.Nr):
            R = self.R_min + dR * idx_r
            R_matrix[idx_r, :] = R
            R_inv_matrix[idx_r, :] = 1 / R
           
        R_integral =  Compute2DIntegral(R_matrix, self.R_min, self.R_max, self.Z_min, self.Z_max)
        R_inv_integral = Compute2DIntegral(R_inv_matrix, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        A1 = (self.mu * self.Ip) **2 * self.beta_p / (8*pi*psi_integral)
        
        if is_scaled:
            A1 /= (self.mu * self.lamda)
        
        A2 = (self.mu * self.Ip + A1 * R_integral) / R_inv_integral
        
        if A1 > 0:
            print("update beta process error : A1 is positive") 
            
        # psi_s로 구한 plamsa current 항 구하기
        psi_n_matrix = np.zeros_like(psi)
        A1_matrix = np.zeros_like(psi)
        
        for idx_r in range(0, self.Nr):
            for idx_z in range(0, self.Nz):
                R = self.R_min + idx_r * dR
                psi_n = (psi[idx_r, idx_z] - self.psi_a) / (self.psi_b - self.psi_a + EPS)     
                psi_n_matrix[idx_r, idx_z] += R * (1 - psi_n ** self.m) ** self.n / self.R_center
                
                A1_matrix[idx_r, idx_z] += R * A1 * (-1)

        psi_n_matrix = np.multiply(psi_n_matrix, self.grid)
        psi_n_integral = Compute2DIntegral(psi_n_matrix, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        A1_matrix = np.multiply(A1_matrix, self.grid)
        A1_integral = Compute2DIntegral(A1_matrix, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        beta_0 = A1_integral / psi_n_integral / self.mu / self.lamda
        
        if abs(beta_0) > 1 or beta_0 <0:
            print("update beta_0 error : beta_0 out of range, value :", beta_0)
            
        self.beta_0 = beta_0
        
    def update_lamda(self, psi):
        
        psi_n_matrix = np.zeros_like(psi)
        psi_a = self.psi_a
        psi_b = self.psi_b
        
        dR = (self.R_max - self.R_min) / (self.Nr - 1)
        Rc = self.R_center
        
        for idx_r in range(0, self.Nr):
            for idx_z in range(0, self.Nz):
                R = self.R_min + idx_r * dR
                psi_n = (psi[idx_r, idx_z] - psi_a) / (psi_b - psi_a + EPS)
                
                psi_n_matrix[idx_r, idx_z] += self.beta_0 * R / Rc * (1 - psi_n ** self.m) ** self.n
                psi_n_matrix[idx_r, idx_z] += (1-self.beta_0) * Rc / R * (1 - psi_n ** self.m) ** self.n
        
        psi_n_matrix = np.multiply(psi_n_matrix, self.grid)
        psi_n_integral = Compute2DIntegral(psi_n_matrix, self.R_min, self.R_max, self.Z_min, self.Z_max)
        
        lamda = self.Ip / psi_n_integral
        self.lamda = lamda
        
    def apply_scaling(self):
        self.psi_a *= self.mu * self.lamda
        self.psi_b *= self.mu * self.lamda