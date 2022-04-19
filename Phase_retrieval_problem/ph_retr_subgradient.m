function [val,val_set] = ph_retr_subgradient(A,b,x,i)
% ============================================================================== %
% Phase retrieval subgradient computation
% ------------------------------------------------------------------------------ %
% ** [val,range] = ph_retr_subgradient(A,b,x,i) **
%
% This function computes the subgradient of the components of the objective  
% function arising in phase retrieval problems. Each component reads:
%
% \[          f_i(x_i) = |(a_i^T x)^2-b_i|$, for $i = 1,...,m$.               \]
%
% INPUT (ordered):
%   • A - matrix, of which each column contains a_i,
%   • b - vector, each row of which contains b_i,
%   • x - subgradient to be evaluated at x,
%   • i - component that needs to be evaluated.
% OUTPUT:
%   • val     - the value of the subgradient if f_i(x_i) is differentiable at $x_i$.
%   • val_set - true and the subgradient set is $val \times [-1,1]$.
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% ______________________________________________________________________________ %
    if (nargin < 4 || isempty(A) || isempty(b) || isempty(x) || isempty(i)) 
        str = '"';
        error("Not enough input arguments.\nType %shelp ph_retr_subgradient%s",str,str);
    end
    if (size(A,2) ~= size(b,1) || size(A,1) ~= size(x,1))
        error('Incorrect input dimensions');
    end
    a = A(:,i);
    tmp = (a'*x)^2 - b(i,1);
    if (tmp == 0)
        val = (2*(a'*x)).*a;
        val_set = true;
    else
        val_set = false;
        val = sign(tmp)*(2*(a'*x)).*a;
    end
end

