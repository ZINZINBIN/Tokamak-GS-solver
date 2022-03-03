# Tokamak Equilibrium Code
## Introduction
Equilibrium Code : Grad-Shafranov Equation Solver
Free Boundary + External Source(PF coils)
## How to Run(Test)
```
conda create env -f environment.yaml
conda activate custom_environment
python3 {filename.py} # (ex : python3 test_free_boundary.py)
```
## Code Structure
```
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

## Reference
- Free boundary Grad-Shafranov Solver : https://github.com/bendudson/freegs
- Derivation and Applications of Grad-Shafranov Equation In Magnetohydrodynamics(MHD) : http://www.questjournals.org/jram/papers/v7-i4/E07043438.pdf
- Topics in Fusion and Plasma Studies : Numerical solution of tokamak equilibriam