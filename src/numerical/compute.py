from matplotlib.pyplot import draw
import numpy as np
from scipy import special
from scipy.linalg import decomp_qr, lu
import math

pi = math.pi
MU = 4 * pi * 10 ** (-7) # permuability for vaccum
EPS = 1e-5
EPS_GRAD = 1e-2
dv_ratio = 1e-2

def GreenFunction(R0, Z0, R, Z, mu = MU):
    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = mu * np.sqrt(R*R0) / 2 / pi / k * ((2-k**2) * ellipK - 2 * ellipE)
    return g

def GreenFunctionScaled(R0, Z0, R, Z):
    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = np.sqrt(R*R0) / 2 / pi / k * ((2-k**2) * ellipK - 2 * ellipE)
    return g

def GreenFunctionMatrix(R, Z, R_min, Z_min, R_max, Z_max, Nr, Nz, mu = MU):
    
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    G = np.zeros((Nr, Nz))
    
    for idx_r in range(0,Nr):
        for idx_z in range(0,Nz):
            R0 = R_min + idx_r * dR
            Z0 = Z_min + idx_z * dZ
            G[idx_r, idx_z] = GreenFunction(R0, Z0, R, Z, mu)      
    return G
 
def GreenFunctionMatrixScaled(R, Z, R_min, Z_min, R_max, Z_max, Nr, Nz):
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    G = np.zeros((Nr, Nz))
    
    for idx_r in range(0,Nr):
        for idx_z in range(0,Nz):
            R0 = R_min + idx_r * dR
            Z0 = Z_min + idx_z * dZ
            G[idx_r, idx_z] = GreenFunctionScaled(R0, Z0, R, Z)      
    return G     
      
def GreenBz(R0, Z0, R, Z, mu, Ic):
    dr = abs(R-R0) * dv_ratio
    Gf = GreenFunction(R0, Z0, R - 0.5 * dr, Z, mu)
    Gi = GreenFunction(R0, Z0, R + 0.5 * dr, Z, mu)
    dGdr = (Gi - Gf) / (dr + EPS)
    Bz = Ic / (R+EPS) * dGdr
    return Bz

def GreenBzScaled(R0, Z0, R, Z, mu, Ic):
    dr = abs(R-R0) * dv_ratio
    Gf = GreenFunctionScaled(R0, Z0, R - 0.5 * dr, Z)
    Gi = GreenFunctionScaled(R0, Z0, R + 0.5 * dr, Z)
    dGdr = (Gi - Gf) / (dr + EPS)
    Bz = 1 / (R+EPS) * dGdr
    scale_factor = mu * Ic
    return Bz, scale_factor

def GreenBr(R0, Z0, R, Z, mu, Ic):
    dz = abs(Z-Z0) * dv_ratio
    Gf = GreenFunction(R0, Z0, R, Z - 0.5 * dz, mu)
    Gi = GreenFunction(R0, Z0, R, Z + 0.5 * dz, mu)
    dGdz = (Gi - Gf) / (dz + EPS)
    Br = (-1) * Ic / (R+EPS) * dGdz
    return Br

def GreenBrScaled(R0, Z0, R, Z, mu, Ic):
    dz = abs(Z-Z0) * dv_ratio
    Gf = GreenFunctionScaled(R0, Z0, R, Z - 0.5 * dz)
    Gi = GreenFunctionScaled(R0, Z0, R, Z + 0.5 * dz)
    dGdz = (Gi - Gf) / (dz + EPS)
    Br = (-1) / (R+EPS) * dGdz
    scale_factor = mu * Ic
    return Br, scale_factor

def GSsolveSOR(A_origin, x_origin, b_origin, w = 1.0):
    '''
    - solve linear algebra for Grad-Shafranov Equation
    - method : SOR(Successive Over Relaxation)
    - A : numpy.ndarray, (m,n), Grad-Shafranov Operator
    - x : numpy.ndarray or numpy.array, (n,) or (n,1), psi
    - b : numpy.ndarray or numpy.array, (m,) or (m,1), Poloidal Current
    - w : Relaxation Factor, w > 1
    '''

    A = np.copy(A_origin)
    x = np.copy(x_origin)
    b = np.copy(b_origin)

    # print("A shape : ", A.shape)
    A_diag = A.diagonal()
    D = np.diag(A_diag)
    p = A - D
    L, U = np.zeros_like(p), np.zeros_like(p)

    rows = D.shape[0]
    cols = D.shape[1]

    for row in range(0, rows):
        for col in range(0, cols):
            if row > col:
                L[row, col] = p[row, col]
            elif row < col:
                U[row, col] = p[row, col]
            else:
                pass
    
    right_term = w * b - (w * U + (w-1) * D) @ x
    
    # SOR alogorithm per 1 epoch
    x_new = np.zeros_like(x)

    for idx in range(0, x_new.shape[0]):
        x_new[idx] = (1-w) * x[idx] + w / (A_diag[idx] + EPS) * (b[idx] - L[idx, :]@x_new - U[idx, :]@x)

    return x_new

def forward_sub(L,b):
    ''' Forward Subsititution Algorithm for solving Lx = b 
    - L : Lower Triangular Matrix, m*n
    - x : n*1
    - b : m*1
    '''
    
    m = L.shape[0]
    n = L.shape[1]
    x_new = np.zeros((n,1))
    
    for idx in range(0,m):
        x_new[idx] = b[idx] - L[idx,:]@x_new
        x_new[idx] /= L[idx,idx]
    
    return x_new

def backward_sub(U,b):
    ''' Backward Subsititution Algorithm for solving Lx = b 
    - L : Lower Triangular Matrix, m*n
    - x : n*1
    - b : m*1
    '''
    m = U.shape[0]
    n = U.shape[1]
    x_new = np.zeros((n,1))
    
    for idx in range(m-1,-1,-1):
        x_new[idx] = b[idx] - U[idx, :]@x_new
        x_new[idx] /= U[idx,idx]
    
    return x_new
    
def GSsolveLinear(A_origin, b_origin, method = 'LU'):
    ''' solve Grad-Shafranov Equation using cholesky factorization
    - A.T @ A => positive definite matrix
    - solve A.T @ A x = A.T b with LR or QR factorization
    - method
    (1) LR factorization
        - L@R = A.T@A
        - inverse matrix of L and R can be computed by converting sign of components
    (2) QR factorization
        - using Gram-슈미트 Method, get orthogonal vector set 
        - Q : matrix [v1, v2, v3, .... vn]
    '''
    
    A = np.copy(A_origin)
    b = np.copy(b_origin)
    
    A_ = A.T@A
    b_ = A.T@b
    
    if method == 'LU':
        P,L,U = lu(A_, overwrite_a = True, check_finite = True)
        
        # pivoting 
        b_ = P.T@b_
        
        # solve LUx = P@b_
        # using Forward and Back Subsititue Algorithm
        # (1) Forward subsitution : LX = B
        # (2) Backward subsitution : UX = B
        
        x_new = backward_sub(U, forward_sub(L, b_))

        return x_new.reshape(-1,)
        
    else:
        return None


def Compute1DIntegral(A_origin, x_min : float, x_max : float):
    
    N = A_origin.shape[0]
    dx = (x_max - x_min) / N
    total = 0
    for idx in range(0,N-1):
        diff = 0.5 * (A_origin[idx] + A_origin[idx + 1]) * dx
        total += diff
        
    return total

def Compute2DIntegral(A_origin, x_min : float, x_max : float, y_min : float, y_max : float):
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / n_mesh_x
    dy = (y_max - y_min) / n_mesh_y 
    
    result = 0
    
    for idx_x in range(0, n_mesh_x - 1):
        for idx_y in range(0, n_mesh_y - 1):
            diff = 0.25 * (A_origin[idx_x, idx_y] + A_origin[idx_x + 1, idx_y] + A_origin[idx_x, idx_y + 1] + A_origin[idx_x + 1, idx_y + 1])
            diff *= dx
            diff *= dy
            result += diff
            
    return result
            
def compute_B_r(psi, R_min, R_max, Z_min, Z_max):
    '''compute B_r from poloidal magnetic flux
    - psi : 2D array with poloidal magnetic flux
    - R_min / R_max / Z_min / Z_max : geometrical components
    - method : FDM 
    '''
    
    Br = np.zeros_like(psi)
    
    Nr = psi.shape[0]
    Nz = psi.shape[1]
    
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0,Nz):
            R = R_min + idx_r * dR
            
            if idx_z == 0:
                d_psi = (-1) * psi[idx_r, idx_z + 2] + 4 * psi[idx_r, idx_z + 1] - 3 * psi[idx_r, idx_z]
                Br[idx_r, idx_z] = (-1) * d_psi / 2 / dZ / R
                
            elif idx_z == Nz - 1 :
                d_psi = (-1) * psi[idx_r, idx_z] + 4 * psi[idx_r, idx_z - 1] - 3 * psi[idx_r, idx_z - 2]
                Br[idx_r, idx_z] = (-1) * d_psi / 2 / dZ / R
                
            else:
                d_psi = psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1]
                Br[idx_r, idx_z] = d_psi / 2 / dZ
                
    return Br

def compute_B_z(psi, R_min, R_max, Z_min, Z_max):
    '''compute B_z from poloidal magnetic flux
    - psi : 2D array with poloidal magnetic flux
    - R_min / R_max / Z_min / Z_max : geometrical components
    - method : FDM 
    '''
    
    Bz = np.zeros_like(psi)
    
    Nr = psi.shape[0]
    Nz = psi.shape[1]
    
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0,Nz):
            R = R_min + idx_r * dR
            
            if idx_r == 0:
                d_psi = (-1) * psi[idx_r + 2, idx_z + 2] + 4 * psi[idx_r + 1, idx_z] - 3 * psi[idx_r, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR / R
                
            elif idx_r == Nr - 1 :
                d_psi = (-1) * psi[idx_r, idx_z] + 4 * psi[idx_r - 1, idx_z] - 3 * psi[idx_r - 2, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR / R
                
            else:
                d_psi = psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR
                
    return Bz

def compute_B_phi(psi, R_min, R_max, Z_min, Z_max):
    
    B_phi = np.zeros_like(psi)
    Nr = psi.shape[0]
    Nz = psi.shape[1]

    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0, Nz):
            R = R_min + idx_r * dR
            Z = Z_min + idx_z * dZ 
    

    return None

def compute_J_phi_plasma(psi : float, psi_a : float, psi_b : float, r : float, r0 : float, lamda : float, beta_0 : float, m : int, n: int):
    '''compute J_phi plasma using fitting curve
    - psi : poloidal magnetic flux / 2pi
    - psi_a : psi on magnetic axis
    - psi_b : psi on plasma boundary(X-point)
    - r : radius from axis to current position
    - r0 : radius form axis to center
    - lamda : coeff(update)
    - beta_0 : coeff(update)
    - m : coeff(fixed)
    - n : coeff(fixed)
    '''
    
    psi_s = (psi - psi_a) / (psi_b - psi_a + EPS)
    
    if lamda is None:
        return (beta_0 * (r/r0) + (1-beta_0) * (r0/r)) * (1-psi_s ** m) ** n
    else:
        return lamda * (beta_0 * (r/r0) + (1-beta_0) * (r0/r)) * (1-psi_s ** m) ** n

def compute_derivative_matrix(A_origin, x_min : float, x_max : float, y_min : float, y_max : float, axis = 0):
    '''compute derivative of matrix dA/dB while B : x or y axis
    - (option) axis : 0 or 1, if 0 then dA/dR, while 1 then dA/dZ
    - x_min, x_max : range of radius
    - y_min, y_max : range of height
    - error order : 2
    '''
    dev_A = np.zeros_like(A_origin)
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / (n_mesh_x - 1)
    dy = (y_max - y_min) / (n_mesh_y - 1)
    
    if axis == 0:
        for idx_x in range(0, n_mesh_x):
            
            if idx_x == 0:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            elif idx_x == n_mesh_x - 1:
                dev = A_origin[idx_x, :] - A_origin[idx_x - 1, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            else:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x - 1, :]
                dev /= (2 * dx)
                dev_A[idx_x, :] = dev
        
    else:
        for idx_y in range(0, n_mesh_y):
            if idx_y == 0:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y]
                dev /= dy
                dev_A[:,idx_y] = dev
            elif idx_y == n_mesh_y - 1:
                dev = A_origin[:, idx_y] - A_origin[:, idx_y - 1]
                dev /= dy
                dev_A[:,idx_y] = dev
            else:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y - 1]
                dev /= (2 * dy)
                dev_A[:,idx_y] = dev
            
    return dev_A

def compute_2nd_derivative_matrix(A_origin, x_min : float, x_max : float, y_min : float, y_max : float, axis = 0):
    '''compute 2nd derivative of matrix d^2A/dx^2, d^2A/dy^2 or d^2A/dxdy
    - (option) axis : 0 or 1, if 0 then dA/dR, while 1 then dA/dZ
    - x_min, x_max : range of radius
    - y_min, y_max : range of height
    - error order : 2
    '''
    dev_A = np.zeros_like(A_origin)
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / (n_mesh_x - 1)
    dy = (y_max - y_min) / (n_mesh_y - 1)
    
    if axis == 0:
        for idx_x in range(0, n_mesh_x):
            
            if idx_x == 0:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            elif idx_x == n_mesh_x - 1:
                dev = A_origin[idx_x, :] - A_origin[idx_x - 1, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            else:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x - 1, :]
                dev /= (2 * dx)
                dev_A[idx_x, :] = dev
        
    else:
        for idx_y in range(0, n_mesh_y):
            if idx_y == 0:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y]
                dev /= dy
                dev_A[:,idx_y] = dev
            elif idx_y == n_mesh_y - 1:
                dev = A_origin[:, idx_y] - A_origin[:, idx_y - 1]
                dev /= dy
                dev_A[:,idx_y] = dev
            else:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y - 1]
                dev /= (2 * dy)
                dev_A[:,idx_y] = dev
            
    return dev_A