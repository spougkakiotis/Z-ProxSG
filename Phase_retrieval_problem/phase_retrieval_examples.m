function [pb_struct] = phase_retrieval_examples(dim,num_of_samples)
% ================================================================================== %
% Phase retrieval problem generator.
% ---------------------------------------------------------------------------------- %
% ** problem_struct = phase_retrieval_examples(dim,num_of_samples) **
%
% This function generates phase retrieval optimization problems over a  
% real Euclidean vector space. The function:
%   • generates m samples $a_i \sim Normal(0,I_d)$, for $i = 1,...,m$,
%     where $d$ is the dimension of the unknown variable,
%   • generates $\bar{x}, x_0$ (target signal, starting point, resp.)
%     over the unit sphere,
%   • and sets $b_i = (a_i^T \bar{x})^2$.
% The optimization problem has the following form:
%
% \[            min_{x \in R^d} f(x) = (1/m) \sum_{i = 1}^{m} f_i(x),             \]
% 
% where $f_i(x) = |(a_i^T x)^2-b_i|$, for $i = 1,...,m$.
%           
%
% INPUT (ordered):
%   • dim - the dimension of the unknown (d above).
%   • num_of_samples - number of sample vector a_i (m above).
% OUTPUT:
%   • pb_struct - a MATLAB struct() with fields:
%       • .f(x,i)           - inline function, evaluating $f(x)$ and 
%                             returning the result, where $x$ is given. 
%       • .sub_f(x,i)       - inline multifunction returning the subgradient 
%                             of f_i(x), given $i$ and $x$.
%       • .prox_f(beta,x,i) - inline function, computing the $prox_{beta*f_{i}}(x)$
%                             with the smallest objective value.
%       • .sample_xi()      - inline function sampling a random component $i$.
%       • .data             - contains all the data, i.e. .A (each column of which 
%                             corresponds to .a_i), .b (each row of which corresponds 
%                             to b_i), .m (number of samples), .d (size of unknown 
%                             variable, start_point (the starting point for 
%                             an arbitrary algorithm).
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% __________________________________________________________________________________ %

    pb_struct = struct();
    pb_struct.data = struct();
    if (nargin < 2 || isempty(dim) || isempty(num_of_samples)) 
        str = '"';
        error("Not enough input arguments.\nType %shelp phase_retrieval_examples%s",str,str);
    end
    pb_struct.data.m = num_of_samples;     
    pb_struct.data.d = dim;
    % ============================================================================== %
    % Generate and store the random data.
    % ------------------------------------------------------------------------------ %
    % Each column of A contains the sample $a_i$, $i = 1,...,m$.
    pb_struct.data.A = randn(dim,num_of_samples); 
    % Sample $\bar{x}$ and $x_0$ uniformly from the d-unit sphere.
    u = randn(dim,1);
    x_bar = u./norm(u);
    u = randn(dim,1);
    pb_struct.data.start_point = u./norm(u);
    pb_struct.data.b = (pb_struct.data.A'*x_bar).^2;
    % ______________________________________________________________________________ %
    
    % ============================================================================== %
    % Functions evaluating $f(x), $\partial f_i(x)$, $prox_{\lambda f_i}(x)$,
    % as well as producing random samples of the stochastic function f().
    % ------------------------------------------------------------------------------ %
    % Computes $f(x)$ = (1/m) \sum_{i = 1}^{m} f_i(x)$
    pb_struct.f = @(x) norm((pb_struct.data.A'*x).^2-pb_struct.data.b,1)/num_of_samples;
    % Computes $f_i(x) = |(a_i^T x)^2-b_i|$.
    pb_struct.f_i = @(x,i) ph_retr_fun_eval(pb_struct.data.A,pb_struct.data.b,x,i);
    % Computes $\partial f_i(x)$.
    pb_struct.sub_f_i = @(x,i) ph_retr_subgradient(pb_struct.data.A,pb_struct.data.b,x,i);
    % Computes the $prox_{\lambda f_i}(x)$ with the smallest objective value.    
    pb_struct.prox_f_i = @(beta,x,i) ph_retr_prox(pb_struct.data.A,pb_struct.data.b,@(y,i) pb_struct.f_i(y,i),beta,x,i);
    % Uniformly draw a component of the stochastic function $f$. 
    pb_struct.sample_xi = @() unidrnd(num_of_samples);
    % ______________________________________________________________________________ %
end

