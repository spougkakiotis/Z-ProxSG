function [reduction,sol_struct] = ProxADMM(sigma,pb_struct,tol,maxit,R_plot)
% ======================================================================================================================== %
% Proximal Alternating Direction Method of Multipliers:
% ------------------------------------------------------------------------------------------------------------------------ %
% [sol_struct] = ProxADMM(Q,D,A,A_tr,b,c,lb,ub,tol,maxit,sigma)
%                                           takes as an input the QP problem data, and applies
%                                           proximal ADMM to find an approximate solution
%                                           of the following primal-dual problem:
%
%            min_{x,w}        c^T x +(1/2)x^T Q x + g(w) + delta_{K}(w),                                    (P)
%            s.t.             Ax = b,   w = x,
%
%            max_{x,y,z}      y_1^T b - (1/2)x^T Q x - delta_K^*(z) - g^*(A^Ty - c - Qx - z),               (D)
%
%                                           where g(x) = \|Dx\|_1,
%                                           with tolerance tol. It terminates after maxit iterations.
%                                           It returns an approximate primal-dual solution (x,y,z).
% INPUT (ordered):
%   • sigma     - the value of the proximal penalty parameter (default = 1e-1).
%   • pb_struct - A struct containing all relevant problem information. Fields:
%       • .Q        - the quadratic part of the Hessian of the primal problem.
%       • .L1_D     - the coefficients of the diagonal scaling within the ell-1 norm in the objective.
%       • .A        - the constraint matrix.
%       • .A_tr     - the transposed constraint matrix.
%       • .b        - the left hand side of the linear constraints.
%       • .c        - the coefficients of the linear part of the objective.
%       • .lb       - the lower bound of the variable x.
%       • .ub       - the upper bound of the variable x.
%       • .n        - number of variables
%       • .m        - number of rows of A
%       • .ob_const - constant term in the objective.
%   • tol      - tolerance for early termination (default = 1e-2).
%   • maxit    - the maximum number of iterations (default = 400).
%   • R_plot   - a logical variable indicating whether to store data relevant for the convergence profiles.    
%
% OUTPUT:
%   • reduction  - the overall residual reduction: 
%                   (res_p(maxit) + res_d(maxit) + compl(maxit))/(res_p(0) + res_d(0) + compl(0)).
%   • sol_struct - a MATLAB struct() with fields:
%       • .x         - approximate primal solution
%       • .y         - approximate dual multipliers
%       • .z         - approximate dual slack
%       • .opt       - true if requested tolerance was reached.
%       
%       
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% ________________________________________________________________________________________________________________________ %
    % ==================================================================================================================== %
    % Initialize parameters and relevant statistics.
    % -------------------------------------------------------------------------------------------------------------------- %
    [m,n] = size(pb_struct.A);
    if (nargin < 3  || isempty(tol))
        tol = 1e-2;
    end
    if (nargin < 4 || isempty(maxit))
        maxit = 400;
    end
    if (nargin < 5 || isempty(R_plot))
        R_plot = false;
    end
    gamma = 1.618;                                                      % ADMM step-length.
    x = zeros(n,1); z = zeros(n,1); y = zeros(m,1); w = zeros(n,1);     % Starting point for ADMM.
    iter = 0;   opt = 0;   
    sol_struct = struct();                                              % Keep all output statistics.
    if (R_plot)                                                           % If true, store and plot relevant data
        sol_struct.res_norm = zeros(maxit,1);
    end
    % ____________________________________________________________________________________________________________________ %
    
    % ==================================================================================================================== %
    % Compute residuals (for termination criteria) and factorize the coefficient matrix of the main pADMM sub-problem.
    % -------------------------------------------------------------------------------------------------------------------- %
    temp_compl = w + z;
    % Proximity operator of g(x) = \|Dx\|_1.
    temp_compl = max(abs(temp_compl)-pb_struct.L1_D, zeros(n,1)).*sign(temp_compl);  
    temp_lb = (temp_compl < pb_struct.lb);
    temp_ub = (temp_compl > pb_struct.ub);
    % Euclidean projection to K.
    temp_compl(temp_lb) = pb_struct.lb(temp_lb);                                  
    temp_compl(temp_ub) = pb_struct.ub(temp_ub);
    % Measure of the complementarity between w and z.
    compl = norm(w -  temp_compl);   
    % Primal residual
    res_p = [pb_struct.b-pb_struct.A*x; w-x];        
    % Dual residual.
    res_d = pb_struct.c+pb_struct.Q*x-pb_struct.A_tr*y+z;                                              
    res_0 = norm(res_p) + norm(res_d) + compl;
    % We implicitly use regularizer R_x = \|Q\|_1 I_n - Off(Q).
    Q_tilde = gamma.*(max(norm(pb_struct.Q,1),1e-4) + spdiags(pb_struct.Q,0));              
    M = [spdiags(-Q_tilde,0,n,n)                 pb_struct.A_tr                                  -speye(n);
          pb_struct.A                     spdiags((1/(sigma*gamma)).*ones(n,1),0,m,m)            sparse(m,n);
          -speye(n)                             sparse(n,m)                 spdiags((1/(sigma*gamma)).*ones(n,1),0,n,n)];   
    [L,D_L,pp] = ldl(M,'lower','vector');
    % ____________________________________________________________________________________________________________________ %

    while(iter < maxit)
        iter = iter+1;
        if (R_plot)
            sol_struct.res_norm(iter) = min((norm(res_p) + norm(res_d) + compl)/res_0,0.9);
        end

        % ================================================================================================================ %
        % Check termination criteria.
        % ---------------------------------------------------------------------------------------------------------------- %
        if (norm(res_p)/(1+norm(pb_struct.b)) < tol && norm(res_d)/(1+norm(pb_struct.c)) < tol &&... 
                                                                compl/(1 + norm(w) + norm(z)) < tol )
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        end
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 1st sub-problem: calculation of w_{j+1} (prox evaluation of g() and then projection to K). 
        % ---------------------------------------------------------------------------------------------------------------- %
        w = x + (1/sigma).*z;
        w = max(abs(w)-(1/sigma).*pb_struct.L1_D, zeros(n,1)).*sign(w);
        w_lb = (w < pb_struct.lb);
        w_ub = (w > pb_struct.ub);
        w(w_lb) = pb_struct.lb(w_lb);
        w(w_ub) = pb_struct.ub(w_ub);
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 2nd sub-problem: calculation of y_{j+1}.
        % ---------------------------------------------------------------------------------------------------------------- %
        rhs = [gamma.*(pb_struct.c + pb_struct.Q*x) - Q_tilde.*x + (1-gamma).*(pb_struct.A_tr*y - z);
               pb_struct.b + (1/(gamma*sigma)).*y;
               -w + (1/(gamma*sigma)).*z];  
        warn_stat = warning;
        warning('off','all');
        lhs = L'\(D_L\(L\(rhs(pp))));
        warning(warn_stat);
        lhs(pp) = lhs;
        x = lhs(1:n);   y = lhs(n+1:n+m);   z = lhs(n+m+1:end); 
        % ________________________________________________________________________________________________________________ %
        
        
        % ================================================================================================================ %
        % Residual Calculation.
        % ---------------------------------------------------------------------------------------------------------------- %       
        temp_compl = w + z;
        temp_compl = max(abs(temp_compl)-pb_struct.L1_D, zeros(n,1)).*sign(temp_compl);
        temp_lb = (temp_compl < pb_struct.lb);
        temp_ub = (temp_compl > pb_struct.ub);
        temp_compl(temp_lb) = pb_struct.lb(temp_lb);
        temp_compl(temp_ub) = pb_struct.ub(temp_ub);
        % Measure of the complementarity between w and z.
        compl = norm(w -  temp_compl);        
        % Primal residual.
        res_p = [pb_struct.b-pb_struct.A*x; w-x];           
        % Dual residual.
        res_d = pb_struct.c+pb_struct.Q*x-pb_struct.A_tr*y+z;                                         
        % ________________________________________________________________________________________________________________ %
    end
    y_2 = z;
    z = retrieve_reformulated_z(pb_struct.L1_D,w,y_2);
    res_maxit = norm(res_p) + norm(res_d) + compl;
    sol_struct.z = z;
    sol_struct.x = x;
    sol_struct.y = y;
    sol_struct.opt = opt;
    reduction = res_maxit/res_0;
end


