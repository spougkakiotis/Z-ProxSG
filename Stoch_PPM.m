function [sol_struct] = Stoch_PPM(prox_F_plus_r,Xi,x_0,T,alpha,plot,F_plot)
% ================================================================================== %
% Stochastic proximal point method.
% ---------------------------------------------------------------------------------- %
% ** [sol_struct] = Stoch_PPM(prox_F_plus_r,Xi,x_0,T,alpha,plot,F_plot) **
%
% This function is a stochastic proximal point method for the 
% solution of composite problems of the form:
%
% \[          min_{x \in R^n}  f(x) + r(x),  f(x) = E[F(x,\xi)]      (P)         \]
% 
% where f() is a stochastic (hopefully) weakly-convex function and r() is a  
% proximable closed convex function. We assume that we can compute the 
% proximal operator of $\alpha(F(x,\xi) + r(x))$ and thus this need to be
% given as an input to the function.  
% The method, at each iteration $j$
%
%   • generates a sample-vector $\xi$ from the (unknown) distribution $P$ 
%     on which $F(x,\xi)$ depends
%   • and evaluates the prox of $\alpha(F(x,\xi)+r(x))$
%
% and upon termination (T+1 iterates) returns the last iteration.
%
%
% INPUT (ordered):
%   • prox_F_plus_r - a function that evaluates the proximity operator 
%                     $prox_{\alpha(F+r)}(x)$ given $x$, $\alpha$, and $\xi$
%                     i.e. call as y = prox_r(x,alpha,xi).
%   • Xi            - returns a vector xi such that F(x,xi) yields a 
%                     realization of the stochastic function $f$.
%   • x_0           - the starting point of the algorithm.
%   • T             - the maximum number of iterations.
%   • alpha         - the step size that is used for the gradient step 
%                     (default $\alpha = 10^{-4}$).
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
    if (nargin < 5 || isempty(alpha))
        alpha = 10^(-4);
    end
    if (nargin < 6 || isempty(plot))
        plot = false;
    end
    if (nargin < 7 || isempty(F_plot))
        F_plot = @(y) y;
    end

    sol_struct = struct();
    % starting point and size of x.
    x = x_0;   
    for t = 0:T
        % Sample a (hopefully i.i.d.) realization of $F(,)$.
        xi = Xi();
        x = prox_F_plus_r(alpha,x,xi);
        % If asked, plot any relevant data depending on $x$ using a user function
        if (plot)
            sol_struct.obj_v(t+1) = F_plot(x);
        end
    end
    % Here we could either sample or in the deterministic case pick the solution.
    % with the best (smallest) objective value.
    sol_struct.x_opt = x;
    sol_struct.opt_it = T;
