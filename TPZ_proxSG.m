function [sol_struct] = TPZ_proxSG(F,prox_r,Xi,mu_1,mu_2,x_0,T,alpha,k,plot,F_plot)
% ================================================================================== %
% Two-point Zeroth-order Proximal Stochastic Gradient method.
% ---------------------------------------------------------------------------------- %
% ** [sol_struct] = TPZ_proxSG(F,prox_r,Xi,mu_1,mu_2,x_0,T,alpha,k,plot,F_plot) **
%
% This function is a zeroth-order proximal stochastic gradient method for the 
% solution of composite problems of the form:
%
% \[          min_{x \in R^n}  f(x) + r(x),  f(x) = E[F(x,\xi)]      (P)         \]
% 
% where f() is a stochastic (hopefully) weakly-convex function and r() is a  
% proximable closed convex function. We assume that we do not have any (sub)- 
% gradient information and only function evaluations of F(,) are available.  
% Thus, given a smoothing parameter $\mu > 0$, we replace (P) by the following 
% surrogate problem
%
% \[       min_{x \in R^n}  f_{\mu_1,\mu_2}(x) + r(x),         (P_{mu})         \]
%
% where $f_{\mu_1,\mu_2}()$ is a smoothed surrogate for $f()$ obtained via the 
% use of two-point Gaussian smoothing. The method, at each iteration $j$
%
%   • generates a sample-vector $\xi$ from the (unknown) distribution $P$ 
%     on which $F(x,\xi)$ depends
%   • generates $2k$ sample-vectors from a standard normal distribution N(0,I_n)
%   • evaluates $F(x+\mu_1 U_i,\xi)$ as well as 
%     $F(x+\mu_1*U_i + \mu_2*U_{i+1},\xi)$, i = 1,3,5..,2k-1
%   • computes an unbiased gradient estimate for $f_{\mu}(), as:
%       $G(x,\xi) = 1/(k*\mu_2)*\sum_i((F(x+\mu_1 U_i+\mu_2 U_{i+1},\xi)
%                                     - F(x + \mu_1 U_i,\xi))U_{i+1})$
%   • performs a proximal gradient step using the unbiased gradient estimate 
%
% and upon termination (T+1 iterates) returns the last iteration.
%
%
% INPUT (ordered):
%   • F      - a function that evaluates F at a given point $x$
%              at a given realization $\xi$; call as y = F(x,xi).
%   • prox_r - a function that evaluates the proximity operator 
%              $prox_{\beta r}(x)$ given $x$ and $\beta$,
%              i.e. call as y = prox_r(x,beta).
%   • Xi     - returns a vector xi such that F(x,xi) yields a 
%              realization of the stochastic function $f$.
%   • mu_1   - the inner smoothing parameter for the Gaussian smoothing.
%   • mu_2   - the outer smoothing parameter for the Gaussian smoothing.
%   • x_0    - the starting point of the algorithm.
%   • T      - the maximum number of iterations.
%   • alpha  - the step size that is used for the gradient step 
%              (default $\alpha = 10^{-4}$).
%   • k      - number of gradient estimates used at each iteration 
%              (default $k = 1$ which suffices for convergence 
%               in theory).
%   • plot   - if true, compute and store all intermediate values of interest.
%   • F_plot - determines what needs to be calculated, given the current iterate,
%              i.e. call as F_plot(x).
%
% OUTPUT:
%   • sol_struct - a MATLAB struct() with fields:
%       • .x_opt  - the approximately optimal solution
%       • .opt_it - the iteration counter producing the returned solution 
%       • .obj_v  - if plot is true, it contains all objective values for each T.
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% __________________________________________________________________________________ %
    if (nargin < 7 || isempty(alpha))
        alpha = 10^(-4);
    end
    if (nargin < 8 || isempty(k) || k <= 0)
        k = 1;
    end
    if (nargin < 9 || isempty(plot))
        plot = false;
    end
    if (nargin < 10 || isempty(F_plot))
        F_plot = @(y) y;
    end
    sol_struct = struct();
    % starting point and size of x.
    x = x_0;
    n = size(x,1);
    % to store all normal random vectors.
    U = zeros(n*(k+1),1);
    if (plot)   % if true, store and plot relevant data
        sol_struct.obj_v = zeros(T+1,1);
    end
   
   
    for t = 0:T
        % Sample a (hopefully i.i.d.) realization of $F(,)$.
        xi = Xi();
        G = zeros(n,1);
        U(1:n,1) = randn(n,1);
        % Evaluate and store the realization $F(x,\xi)$.
        F_t = F(x+mu_1.*U(1:n,1),xi);
        for j = 1:k
            % (almost) i.i.d. uniform random vectors.
            U(j*n+1:(j+1)*n,1) = randn(n,1);
            % sum of unbiased gradient estimates for $f_{\mu}()$.
            G = G + ((1/mu_2)*(F(x+ mu_1.*U(1:n,1) ... 
                                + mu_2.*U(j*n+1:(j+1)*n,1),xi)...
                              -F_t)).*U(j*n+1:(j+1)*n,1);
        end
        % Averaging the sum to obtain the gradient estimate.
        G = G./k;
        x = prox_r(x-alpha.*G,alpha);
        % If asked, plot any relevant data depending on $x$ using a user function
        if (plot)
            sol_struct.obj_v(t+1) = F_plot(x);
        end
    end
    % Here we could either sample or in the deterministic case pick the solution.
    % with the best (smallest) objective value.
    sol_struct.x_opt = x;
    sol_struct.opt_it = T;
