import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Dict, Optional, Literal
from tqdm import tqdm

class ElipticOperator:
    def __init__(self, Rl:float, Rr:float, Zl:float, Zr:float, Nr : int, Nz : int, order:Literal['2nd', '4th'] = '4th'):
        self.Rl = Rl
        self.Rr = Rr
        self.Zl = Zl
        self.Zr = Zr
        
        self.Nr = Nr
        self.Nz = Nz
        
        self.order = order
        self.operator = None
        
        # coefficient for 4th order
        self.centred_1st = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]

        self.offset_1st = [
            (-1, -3.0 / 12),
            (0, -10.0 / 12),
            (1, 18.0 / 12),
            (2, -6.0 / 12),
            (3, 1.0 / 12),
        ]

        # Coefficients for second derivatives
        # (index offset, weight)
        self.centred_2nd = [
            (-2, -1.0 / 12),
            (-1, 16.0 / 12),
            (0, -30.0 / 12),
            (1, 16.0 / 12),
            (2, -1.0 / 12),
        ]

        self.offset_2nd = [
            (-1, 10.0 / 12),
            (0, -15.0 / 12),
            (1, -4.0 / 12),
            (2, 14.0 / 12),
            (3, -6.0 / 12),
            (4, 1.0 / 12),
        ]
        
        self.generate_operator(Nr,Nz,order)
    
    def generate_operator(self, Nr, Nz, order:Literal['2nd', '4th'] = '4th'):
        
        if self.Nr == Nr and self.Nz == Nz and self.order == order:
            return

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)

        N_total = Nr * Nz
        
        invdr = 1.0 / dr
        invdr2 = 1.0 / dr ** 2
        
        invdz = 1.0 / dz
        invdz2 = 1.0 / dz ** 2

        A = np.zeros((N_total, N_total))
        
        if self.order == '2nd':     
            for idx_r in range(1, Nr - 1):
                for idx_z in range(1, Nz - 1):
                    R = self.Rl + dr * idx_r
                    row = Nz * idx_r + idx_z
                    drow = Nz
                    A[row, row - 1] = invdz2
                    A[row, row + 1] = invdz2
                    A[row, row] = -2.0 * (invdr2 + invdz2)
                    A[row, row - drow] = (invdr2 + invdr / 2 / R)
                    A[row, row + drow] = (invdr2 - invdr / 2 / R) 
        else:
            for idx_r in range(1, Nr - 1):
                for idx_z in range(1, Nz - 1):
                    R = self.Rl + dr * idx_r
                    row = Nz * idx_r + idx_z
                    drow = Nz
                    
                    if idx_z == 1:
                        for offset, weight in self.offset_2nd:
                            A[row, row + offset] += weight * invdz2
                    elif idx_z == Nz - 2:
                        for offset, weight in self.offset_2nd:
                            A[row, row - offset] += weight * invdz2
                    else:
                        for offset, weight in self.centred_2nd:
                            A[row, row + offset] += weight * indvz2
                        
                    if idx_r == 1;
                        for offset, weight in self.offset_2nd:
                            A[row, row + offset * Nz] += weight * invdr2
                        for offset, weight in self.offset_1st:
                            A[row, row + offset * Nz] -= weight * invdr1 / R
                    elif idx_r == nr - 2:
                        for offset, weight in self.offset_2nd:
                            A[row, row - offset * Nz] += weight * invdr2
                        for offset, weight in self.offset_1st:
                            A[row, row - offset * Nz] += weight * invdr1 / R
                    else:
                        for offset, weight in self.centred_2nd:
                            A[row, row + offset * Nz] += weight * invdr2
                        for offset, weight in self.centred_1st:
                            A[row, row + offset * Nz] -= weight * invdr1 / R
                            
            # boundary
            for idx_r in range(Nr):
                for idx_z in [0,Nz-1]:
                    row = idx_r * Nz + idx_z
                    A[row,row] = 1.0
            
            for idx_r in [0, Nr - 1]:
                for idx_z in range(Nz):
                    row = idx_r * Nz + idx_z
                    A[row,row] = 1.0
                    
        self.operator = A
    
    def get_matrix(self):
        return self.operator

    def __call__(self, psi : np.ndarray):
        
        if psi.ndim == 2:
            psi = psi.reshape(-1,1) 
        
        return np.matmul(self.operator, psi).reshape(self.Nr, self.Nz)


class GSsolver(object):
    def __init__(self, 
                 properties,
                 source,
                 R_min : float, 
                 R_max : float, 
                 Z_min : float, 
                 Z_max : float, 
                 Nr : int, 
                 Nz : int, 
                 convergence : float = 1e-5,
                 verbose_log : int = 8
                 ):
        
        self.properties = properties
        self.source = source
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.Nr = Nr
        self.Nz = Nz
        self.err = None
        self.convergence = convergence
        self.verbose_log = verbose_log
        
        if type(self.source) == dict:
            self.source_upper = self.source["PFcoils_upper"]
            self.source_lower = self.source["PFcoils_lower"]
        else:
            self.source_upper = None
            self.source_lower = None
            
        # 외부 source에서 생성된 psi가 초기값
        if self.source_upper is not None:
            self.psi = self.source_upper.get_psi_boundary_from_coil() + self.source_lower.get_psi_boundary_from_coil()
        else: 
            self.psi = self.source.get_psi_boundary_from_coil()
        
        # plasma boundary 및 grid 초기화(생성) : psi_a, psi_b, grid 초기화
        self.properties.update_plasma_boundary(self.psi)
        
        # beta_0 and lamda 값 초기화
        self.properties.update_lamda(self.psi)
        self.properties.update_beta_0(self.psi)
        
        # J_phi_plasma 초기화
        self.J_phi = self.properties.get_J_phi_plasma(self.psi)
    
        # psi_boundary 생성
        if self.source_upper is not None:
            psi_boundary_coil = self.source_lower.get_psi_boundary_from_coil() + self.source_upper.get_psi_boundary_from_coil()
        else:
            psi_boundary_coil = self.source.get_psi_boundary_from_coil()
        
        # psi_boundary_coil은 1회 계산 이후 compute 하는 과정에서 저장된 값만 활용(1회 초기화 이후 계산 x)
        self.psi_boundary_coil = psi_boundary_coil  
        
        # psi_boundary_plasma = self.properties.get_psi_boundary_from_plasma(self.psi)
        # self.psi_boundary = psi_boundary_coil + psi_boundary_plasma
        self.psi_boundary = psi_boundary_coil
        self.psi_boundary = np.multiply(self.psi_boundary, self.properties.grid)
        
        # psi initialization : psi_boundary(source)로 최소 생성
        self.psi = self.psi_boundary
        
        # Grad-Shafranov Operator Matrix 생성
        mat = GSMatrix(self.R_min, self.R_max, self.Z_min, self.Z_max)
        self.GSMatrix = mat(self.Nr, self.Nz)
        np.fill_diagonal(self.GSMatrix, -1)
        
    def solve(self, w = 1.2, iteration = 64):
        
        GS_A = self.GSMatrix
        dr = (self.R_max - self.R_min) / (self.Nr - 1)
        dz = (self.Z_max - self.Z_min) / (self.Nz - 1)

        a = dr**2 * dz**2 / (dr**2 + dz**2)
        
        print("# GS solver operated....")
        
        # Picard Iteration
        for n_iter in tqdm(range(iteration)):
            
            psi_1d = self.psi.reshape(-1,1)
            J_phi_1d = self.J_phi.reshape(-1,1)
            psi_boundary_1d = self.psi_boundary.reshape(-1,1)
            psi_1d_new = np.zeros_like(self.psi)
            
            for idx_r in range(0,self.Nr):
                for idx_z in range(0,self.Nz):
                    R = self.R_min + dr * idx_r
                    row = self.Nz * idx_r + idx_z
                    J_phi_1d[row] *= a * self.properties.mu * R 
            
            # SOR Algorithm to update new psi
            psi_1d_new = GSsolveSOR(GS_A, psi_boundary_1d, J_phi_1d, w)
            self.err = np.linalg.norm((psi_1d_new - psi_1d)/(psi_1d + EPS)) 
            
            # update psi
            psi_1d = psi_1d_new
            self.psi = psi_1d.reshape(self.Nr, self.Nz)
            
            # update parameter
            self.properties.update_plasma_boundary(self.psi)
            self.properties.update_lamda(self.psi)
            self.properties.update_beta_0(self.psi)
            
            # update J_phi, psi_boundary
            self.J_phi = self.properties.get_J_phi_plasma(self.psi)
            self.psi_boundary = self.properties.get_psi_boundary_from_plasma(self.psi) + self.psi_boundary_coil
            self.psi_boundary = np.multiply(self.psi_boundary, self.properties.grid)
            
            # logging
            if n_iter % self.verbose_log == 0:
                print("# GS SOR algorithm, n_iter : {}, relative err : {:.6f}".format(n_iter, self.err))
            
            # check if converged
            if self.err <= self.convergence:
                print("Grad-Shafranov Equation Converged...!")
                break
        
        # check if not converged
        if self.err > self.convergence:
                print("Grad-Shafranov Equation Not Converged...!")
                
    def plotGrid(self, shell_number = 8, save_dir = "./results/Free-Boundary-Equilibrium.png"):
    
        dr = (self.R_max - self.R_min) / (self.Nr)
        dz = (self.Z_max - self.Z_min) / (self.Nz)
        
        x = np.arange(self.R_min, self.R_max, dr)
        y = np.arange(self.Z_min, self.Z_max, dz)

        x,y = np.meshgrid(x, y)

        z_psi = self.psi
        z_J_phi = self.J_phi

        psi_min = np.min(z_psi)
        psi_max = np.max(z_psi)

        J_phi_min = np.min(z_J_phi)
        J_phi_max = np.max(z_J_phi)

        d_psi = (psi_max - psi_min) / shell_number
        d_J_phi = (J_phi_max - J_phi_min) / shell_number
        
        # fig = plt.figure(figsize = (8, 14))
        # gs = gridspec.GridSpec(nrows = 1, ncols = 2, height_ratios = [8], width_ratios = [7,7])
        # ax0 = plt.subplot(gs[0])
        
        plt.figure(figsize = (14,8))
        plt.subplot(1,2,1)
        cs_psi = plt.contour(x,y,z_psi,levels = np.arange(psi_min, psi_max, d_psi))
        plt.xlabel("R-radius")
        plt.ylabel("Z-height")
        plt.title("Free-Boundary Psi : Poloidal Magnetic Flux")
        plt.colorbar()

        plt.subplot(1,2,2)
        cs_J_phi = plt.contour(x,y,z_J_phi,levels = np.arange(J_phi_min, J_phi_max, d_J_phi))
        plt.xlabel("R-radius")
        plt.ylabel("Z-height")
        plt.title("Free-Boundary J_phi : Toroidal Plasma Current")
        plt.colorbar()

        plt.savefig(save_dir)

class GSsolverFreeBoundary(object):
    def __init__(self, Rl, Rr, Zl, Zr, m, n, beta_0, lamda_0, mu = MU, w  = 1.0, source : Dict[str, float] = None):
        self.m = m
        self.n = n
        self.beta_0 = beta_0
        self.lamda_0 = lamda_0
        self.source = source
        self.Rl = Rl
        self.Rr = Rr
        self.Zl = Zl
        self.Zr = Zr
        self.mu = mu
        self.gs_matrix = GSMatrix(Rl,Rr,Zl,Zr)

        self.l2_err = 0
        self.w = w # relaxation factor for solving GS equation

        Ic = self.source["PFcoil_current"]
        rc = self.source["rc"]
        zc = self.source["zc"]

        self.Ic = Ic
        self.Rc = rc
        self.Zc = zc

        self.R0 = (Rl + Rr) / 2
        self.Z0 = (Zl + Zr) / 2

    def initializeGrid(self, Nr, Nz):
        # for free boundary
        psi = np.zeros((Nr,Nz))
        J_phi = np.zeros((Nr,Nz))

        Ic = self.Ic
        rc = self.Rc
        zc = self.Zc

        r0 = self.R0
        z0 = self.Z0

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)

        dR = dv_ratio * dr
        dZ = dv_ratio * dz

        # Initialize psi and J_phi
        for idx_row in range(0, Nr):
            for idx_col in range(0, Nz):
                R = self.Rl + dr * idx_row
                Z = self.Zl + dz * idx_col

                psi[idx_row, idx_col] = Ic * GreenFunction(rc,zc,R,Z)   
                
                dBrdz = (GreenBr(rc,zc, R, Z + 0.5 * dZ, self.mu, Ic) - GreenBr(rc,zc, R, Z - 0.5 * dZ, self.mu, Ic)) / dZ
                dBzdr = (GreenBr(rc,zc, R + 0.5 * dR, Z, self.mu, Ic) - GreenBr(rc,zc, R - 0.5 * dR, Z, self.mu, Ic)) / dR

                J_phi[idx_row, idx_col] = 1 / self.mu * (dBrdz - dBzdr)

        self.psi = psi
        self.J_phi = J_phi

        # Initialize psi_a, psi_b (B.C condition)
        psi_a = np.zeros_like(psi)
        psi_b = np.zeros_like(psi)

        for idx_row in range(0, Nr):
            for idx_col in range(0, Nz):

                R = self.Rl + dr * idx_row
                Z = self.Zl + dz * idx_col

                if idx_col == 0 or idx_row == 0:
                    psi_b[idx_row, idx_col] += Ic * GreenFunction(rc,zc,R,Z) + self.getJpBoundary(J_phi, R, Z)
                    
        psi_b = np.sum(psi_b[np.where(psi_b > 0)]) / 2 / (Nr + Nz)
        psi_a = 0

        self.psi_b = psi_b
        self.psi_a = psi_a
                    
        # get Grad-Shafranov Operator(A)
        self.A = self.gs_matrix(Nr, Nz)
        
    def getJpBoundary(self, J_phi, Rb, Zb):
        
        Nr = J_phi.shape[0]
        Nz = J_phi.shape[1]

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)

        Jp_boundary = 0

        for idx_r in range(0,Nr):
            for idx_z in range(0,Nz):
                R = self.Rl + dr * idx_r
                Z = self.Zl + dz * idx_z
                
                Jp_boundary += GreenFunction(Rb, Zb, R, Z) * J_phi[idx_r, idx_z] * dr * dz

        return Jp_boundary

    def updateJ_phi(self):

        psi_a = self.psi_a
        psi_b = self.psi_b
        psi = self.psi

        r0 = self.R0
        z0 = self.Z0

        Nr = psi.shape[0]
        Nz = psi.shape[1]

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)

        for idx_r in range(0, Nr):
            for idx_z in range(0, Nz):

                R = self.Rl + dr * idx_r
                Z = self.Zl + dz * idx_z

                psi_rz = psi[idx_r, idx_z]
                psi_s = (psi_rz - self.psi_a) / (self.psi_b - self.psi_a)
                self.J_phi[idx_r, idx_z] = self.lamda_0 * (self.beta_0 * R / self.R0 + (1-self.beta_0) * self.R0 / R) * (1-psi_s ** self.m) ** self.n
        

    def updateBoundary(self):
        
        psi = self.psi
        psi_b = np.zeros_like(psi)

        Nr = psi.shape[0]
        Nz = psi.shape[1]

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)
        
        for idx_row in range(0, Nr):
            for idx_col in range(0, Nz):

                R = self.Rl + dr * idx_row
                Z = self.Zl + dz * idx_col

                if idx_col == 0 or idx_row == 0:
                    psi_b[idx_row, idx_col] += self.Ic * GreenFunction(self.Rc,self.Zc,R,Z) + self.getJpBoundary(self.J_phi, R, Z)
                    
        psi_b = np.sum(psi_b[np.where(psi_b > 0)]) / 2 / (Nr + Nz)
        self.psi_b = psi_b

    def updatePsi(self):
        # solve Grad-Shafranov Equation using Picard Iteration
        psi_1d = self.psi.reshape(-1,)
        J_phi_1d = self.J_phi.reshape(-1,)

        Nr = self.psi.shape[0]
        Nz = self.psi.shape[1]
        N_total = Nr * Nz
        
        GS_A = self.gs_matrix(Nr,Nz)
        
        np.fill_diagonal(GS_A, -1)

        dr = (self.Rr - self.Rl) / (Nr - 1)
        dz = (self.Zr - self.Zl) / (Nz - 1)

        a = dr**2 * dz**2 / (dr**2 + dz**2)

        for idx_r in range(0,Nr):
            for idx_z in range(0,Nz):
                R = self.Rl + dr * idx_r
                row = Nz * idx_r + idx_z
                J_phi_1d[row] *= a * self.mu * R 
                
        # SOR method 
        psi_1d_new = GSsolveSOR(GS_A, psi_1d, J_phi_1d, w = 1.0)
        
        # Cholesky Factorization
        # psi_1d_new = GSsolveLinear(GS_A, J_phi_1d)
        
        # get error term from l2 norm
        self.l2_err = np.linalg.norm((psi_1d_new - psi_1d))

        psi_1d_new = psi_1d + self.w * (psi_1d_new - psi_1d)
        self.psi = psi_1d_new.reshape(Nr, Nz)
        

    def solve(self, Nr : int = 64, Nz : int = 64, iteration : int = 128, convergence : float = 1e-8, verbose_log : int = 8):

        assert Nr > 1, "Nr should be larger than 1"
        assert Nz > 1, "Nz should be larger than 1"
        assert convergence < 1, "convergence condition should be smaller than 1"

        self.Nr = Nr
        self.Nz = Nz

        # initalize grid
        self.initializeGrid(Nr, Nz)
        
        # update Boundary value of J_phi and psi_b
        self.updateBoundary()

        # Picard Iteration
        for n_iter in range(0,iteration):

            # update psi by solving gs equation
            self.updatePsi()
            
            # update J_phi from fitted equation
            self.updateJ_phi()

            # update boundary condition 
            self.updateBoundary()
            
            if n_iter % verbose_log == 0:
                print("# n_iter : {}, convergence error : {:.8f}".format(n_iter, self.l2_err))
            
            # check if converged
            if self.l2_err < convergence:
                break

        if self.l2_err < convergence:
            print("GS equilibrium solution converged.....!")
        else:
            print("GS equilibrium solution Not Converged.....!")


    def plotGrid(self, shell_number = 8, save_dir = "./results/Free-Boundary-Equilibrium.png"):

        dr = (self.Rr - self.Rl) / (self.Nr)
        dz = (self.Zr - self.Zl) / (self.Nz)
        
        x = np.arange(self.Rl, self.Rr, dr)
        y = np.arange(self.Zl, self.Zr, dz)

        x,y = np.meshgrid(x, y)

        z_psi = self.psi
        z_J_phi = self.J_phi

        psi_min = np.min(z_psi)
        psi_max = np.max(z_psi)

        J_phi_min = np.min(z_J_phi)
        J_phi_max = np.max(z_J_phi)

        d_psi = (psi_max - psi_min) / shell_number
        d_J_phi = (J_phi_max - J_phi_min) / shell_number
        
        # fig = plt.figure(figsize = (8, 14))
        # gs = gridspec.GridSpec(nrows = 1, ncols = 2, height_ratios = [8], width_ratios = [7,7])
        # ax0 = plt.subplot(gs[0])
        
        plt.figure(figsize = (14,8))
        plt.subplot(1,2,1)
        cs_psi = plt.contour(x,y,z_psi,levels = np.arange(psi_min, psi_max, d_psi))
        plt.xlabel("R-radius")
        plt.ylabel("Z-height")
        plt.title("Free-Boundary Psi : Poloidal Magnetic Flux")
        plt.colorbar()

        plt.subplot(1,2,2)
        cs_J_phi = plt.contour(x,y,z_J_phi,levels = np.arange(J_phi_min, J_phi_max, d_J_phi))
        plt.xlabel("R-radius")
        plt.ylabel("Z-height")
        plt.title("Free-Boundary J_phi : Toroidal Plasma Current")
        plt.colorbar()

        plt.savefig(save_dir)
        




    
    