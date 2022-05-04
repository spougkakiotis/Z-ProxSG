# Z-ProxSG
A zeroth-order proximal stochastic gradient method for non-smooth and non-convex optimization

This file is used to derive the numerical results in the following paper:

A Zeroth-order Proximal Stochastic Gradient Method for Weakly Convex Stochastic Optimization  
Spyridon Pougkakiotis, Dionysios S. Kalogerias, https://arxiv.org/abs/2205.01633

It contains two zeroth-order proximal stochastic gradient methods, a proximal stochastic 
sub-gradient method, a stochastic proximal point method, as well as a proximal alternating
direction method of multipliers suitable for the solution of ℓ1-regularized convex 
quadratic programming. 

Two experiments are set up:
  1) The solution of randomly generated phase retrieval instances (weakly convex problems)
  2) The hyper-parameter tuning of the proximal ADMM for the solution of L1-regularized
     PDE-constrained optimization problems.

Each file is heavily commented for the convenience of the user. Use help "fuction of interest" 
to learn more about each of the included files.

The proximal ADMM and the PDE-problems solved in the second experiment are taken from the 
following work:

A semismooth Newton-proximal method of multipliers for ℓ1-regularized convex quadratic programming         
Spyridon Pougkakiotis, Jacek Gondzio,  https://doi.org/10.48550/arXiv.2201.10211
