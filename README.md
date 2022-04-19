# Z-ProxSG
A zeroth-order proximal stochastic gradient method for non-smooth and non-convex optimization

This file is used to derive the numerical results in the following paper:


It contains two zeroth-order proximal stochastic gradient methods, a proximal stochastic 
sub-gradient method, a stochastic proximal point method, as well as a proximal alternating
direction method of multipliers suitable for the solution of $\ell_1$-regularized convex 
quadratic programming. 

Two experiments are set up:
  1) The solution of randomly generated phase retrieval instances (weakly convex problems)
  2) The hyper-parameter tuning of the proximal ADMM for the solution of $L^1$-regularized,
     PDE-constrained optimization problems.

Each file is heavily commented for the convenience of the user. Use help "fuction of interest" 
to learn more about each of the included files.

The proximal ADMM and the PDE-problems solved in the second experiment are taken from the 
following work:

A semismooth Newton-proximal method of multipliers for ℓ1-regularized convex quadratic programming
              Spyridon Pougkakiotis, Jacek Gondzio
              https://doi.org/10.48550/arXiv.2201.10211
