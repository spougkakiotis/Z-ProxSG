function [opt] = ph_retr_prox(A,b,f,beta,x,i)
% ============================================================================== %
% Phase retrieval prox computation
% ------------------------------------------------------------------------------ %
% ** [opt] = ph_retr_prox(A,b,x,i) **
%
% This function computes the prox operator of the components of the objective  
% function arising in phase retrieval problems. Each component reads:
%
% \[            f_i(x) = |(a_i^T x)^2-b_i|$, for $i = 1,...,m$.               \]
%
% and the function computes all $x^*$ such that
%
% \[  x^* = \arg\min_y \{f_i(y) + (1/(2*beta)) \|y-x\|_2^2\}.          (P)    \]
%
% INPUT (ordered):
%   • A    - matrix, of which each column contains a_i,
%   • b    - vector, each row of which contains b_i,
%   • f    - a functional evaluating the objective function at a given point,
%   • beta - the proximal penalty parameter,
%   • x    - prox to be evaluated at x,
%   • i    - component that needs to be evaluated.
% OUTPUT:
%   • opt - returns the prox point with the smallest function value, $f_i(x)$. 
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% ______________________________________________________________________________ %

    % ========================================================================== %
    % Problem (P) admits 4 closed-form solutions. Store them and find the best.
    % -------------------------------------------------------------------------- %
    n = size(x,1);
    y = zeros(4*n,1);
    a = A(:,i);
    tmp = a'*x;
    y(1:n,1)       = x - ((2*beta*tmp)/(2*beta*norm(a)^2 + 1)).*a;
    y(n+1:2*n,1)   = x - ((2*beta*tmp)/(2*beta*norm(a)^2 - 1)).*a;
    y(2*n+1:3*n,1) = x - ((tmp + sqrt(b(i,1)))/(norm(a)^2)).*a;
    y(3*n+1:4*n,1) = x - ((tmp - sqrt(b(i,1)))/(norm(a)^2)).*a;
    f_opt = Inf;
    for j = 0:3
        y_j = y(j*n+1:(j+1)*n,1); 
        f_tmp = f(y_j,i) + (1/(2*beta))*norm(y_j-x)^2;
        if (f_tmp < f_opt)
            opt = y_j;
            f_opt = f_tmp;
        end
    end
    % __________________________________________________________________________ %


end

