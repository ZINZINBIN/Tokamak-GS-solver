# Tokamak Equilibrium Code
## Introduction
This is a github repository of python code for solving Grad-Shafranov equation for describing tokamak plasma equilibrium state. We used two approaches for solving GS equation as given below.
- Numerical computation code for solving Grad-Shafranov equation (based on FreeGS)
- Physics informend neural networks (PINN)
In this code, we assumed that PFC coil currents are given and used additional constraints such as betap or Ip. The input data for PINN can be 0D parameters for tokamak plasma such as betap, q95, Ip, internal inductance and so on. 

<div>
    <p float = 'left'>
        <img src="/result/PINN_profile.png"  width="640" height="240">
    </p>
</div>

We solved Grad-Shafranov equation for a free-boundary case. Especially, we constructed our PINN-based GS solver for a free-boundary case. The main reference for our code is <a href = "https://arxiv.org/abs/1503.03135" target = "_blank">[Paper : Development of a Free boundary Tokamak Equilibrium Solver for Advanced Study of Tokamak Equilibria, Young Mu Jeon]</a>. 

<div>
    <p float = 'left'>
        <img src="/result/PINN_psi_norm.png"  width="480" height="240">
    </p>
</div>

## How to run
- Setting : environment
    ```
    conda create env -f environment.yaml
    conda activate custom_environment
    ```
- Execute training code for PINN based GS solver
    ```
    # training PINN based GS solver from KSTAR EFIT dataset
    python3 train_PINN.py --gpu_num {gpu_num} --tag {tag} --save_dir {directory for saving results}
    ```
- Execute computing code for free boundary GS solver
    ```
    # test code for solving free boundary problem 
    python3 test_free_boundary.py --iteration {iteration number} --shell_number {contour boundary number}
                                  --save_dir {directory for saving results}
    ```

## Detail
### process
1. Initialize Grid, $\psi_{k}$
2. Update $J_{\phi}(\psi_{k})$
3. Solve Grad-Shafranov Equation to get new $\psi_{k}$
4. If converged, post-processing

### GS solver
1. Need Boundary Condition, $\psi_{boundary}$ to get $\psi$
2. update $\beta_{0}$ and $\lambda$ from constraints
3. update $J_{\phi}$ and repeat 1

### PINN based GS solver
1. Forward process of the neural networks : computation of the output with given input data(ex : plasma state information)
2. Computation of the Physics-informed loss (ex : Grad-shafranov equation) and constraint loss (ex : total plasma current from GS solver = Ip)
3. Backward process: update the parameters of the networks

## Reference
- Free boundary Grad-Shafranov Solver : https://github.com/bendudson/freegs
- Derivation and Applications of Grad-Shafranov Equation In Magnetohydrodynamics(MHD) : http://www.questjournals.org/jram/papers/v7-i4/E07043438.pdf
- Topics in Fusion and Plasma Studies : Numerical solution of tokamak equilibriam