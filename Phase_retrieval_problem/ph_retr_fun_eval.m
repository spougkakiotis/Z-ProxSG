function [val] = ph_retr_fun_eval(A,b,x,i)
% ============================================================================== %
% Phase retrieval sample objective function evaluation, $i \in {1,...,m}$.
% ------------------------------------------------------------------------------ %
% ** [val,range] = ph_retr_fun_eval(A,b,x,i)
%
% This function computes the subgradient of the components of the objective  
% function arising in phase retrieval problems. Each component reads:
%
% \[          f_i(x_i) = |(a_i^T x)^2-b_i|$, for $i = 1,...,m$.               \]
%
% INPUT (ordered):
%   • A - matrix, of which each column contains a_i,
%   • b - vector, each row of which contains b_i,
%   • x - stochastic function to be evaluated at x,
%   • i - stochastic sample $i \in {1,...,m}$.
% OUTPUT:
%   • val - the value of f_i(x_i) at $x_i$.
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% ______________________________________________________________________________ %
    val = abs((A(:,i)'*x)^2 - b(i,1));
end

