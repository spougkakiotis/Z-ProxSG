function [sol_struct] = ProxSSG(sub_F,prox_r,Xi,x_0,T,alpha,plot,F_plot)
% ================================================================================== %
% Proximal Stochastic Sub-Gradient method.
% ---------------------------------------------------------------------------------- %
% ** [sol_struct] = ProxSSG(sub_F,prox_r,Xi,x_0,T,alpha,plot,F_plot) **
%
% This function is a proximal stochastic sub-gradient method for the 
% solution of composite problems of the form:
%
% \[          min_{x \in R^n}  f(x) + r(x),  f(x) = E[F(x,\xi)]      (P)         \]
% 
% where f() is a stochastic (hopefully) weakly-convex function and r() is a  
% proximable closed convex function. We assume that we can use sub-gradient 
% information and function evaluations of F(,), which are given.  
% The method, at each iteration $j$
%
%   • generates a sample-vector $\xi$ from the (unknown) distribution $P$ 
%     on which $F(x,\xi)$ depends
%   • evaluates the sub-gradient of $F(x,\xi)$
%   • performs a proximal gradient step via the unbiased sub-gradient estimate 
%
% and upon termination (T+1 iterates) returns the last iteration.
%
%
% INPUT (ordered):
%   • prox_r - a function that evaluates the proximity operator 
%              $prox_{\beta r}(x)$ given $x$ and $\beta$,
%              i.e. call as y = prox_r(x,beta).
%   • sub_F  - the sub-gradient of F, given $x$ and $\xi$;
%              call as [y,val_set] = sub_F(x,xi), where
%              if val_set = true, then the subgradient 
%              is the set [-y,y].
%   • Xi     - returns a vector xi such that F(x,xi) yields a 
%              realization of the stochastic function $f$.
%   • x_0    - the starting point of the algorithm.
%   • T      - the maximum number of iterations.
%   • alpha  - the step size that is used for the gradient step 
%              (default $\alpha = 10^{-4}$).
%   • plot   - if true, compute and store all intermediate values of interest.
%   • F_plot - determines what needs to be calculated, given the current iterate,
%              i.e. call as F_plot(x).
%
% OUTPUT:
%   • sol_struct - a MATLAB struct() with fields:
%       • .x_opt  - the approximately optimal solution
%       • .opt_it - the iteration counter producing the returned solution   
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% __________________________________________________________________________________ %
    if (nargin < 6 || isempty(alpha))
        alpha = 10^(-4);
    end
    if (nargin < 7 || isempty(plot))
        plot = false;
    end
    if (nargin < 8 || isempty(F_plot))
        F_plot = @(y) y;
    end

    sol_struct = struct();
    % starting point and size of x.
    x = x_0;   
    for t = 0:T
        % Sample a (hopefully i.i.d.) realization of $F(,)$.
        xi = Xi();
        [G,~] = sub_F(x,xi);
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
