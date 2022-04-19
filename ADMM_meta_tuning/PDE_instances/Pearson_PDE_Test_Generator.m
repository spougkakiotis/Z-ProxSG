function [pb_struct] = Pearson_PDE_Test_Generator(problem_choices,DS)
% ====================================================================================================== %
% This function takes an integer as an input, which specifies the problem 
% that is being solved by SSN-PMM. Then, it generates and return the relevant data 
% required to solve the problem, in the form of a structure.
% ------------------------------------------------------------------------------------------------------ %
% Input
%       • problem_choices: 1 -> Poisson Control with L^1/L^2-regularization and control bounds     
%                          2 -> Convection Diffusion with L^1/L^2-regularization and control bounds
%       • DS: a data struct, containing potential values for the L2 and L1 regularization parameters,
%             as well as for the discretization size, from which one should uniformly sample 
%             and create the problem instance.
%                        
% Output
%       • pb_struct: A structure of structures containing all relevant problem data.
% ______________________________________________________________________________________________________ %
    
    grid_type = 1; % uniform grid
    % ================================================================================================== %
    % Set the problem parameters and data.
    % -------------------------------------------------------------------------------------------------- %
    % Sample uniformly indexes for the data matrices in DS
    i_alpha_1 = unidrnd(size(DS.alpha_1_vals,1));
    i_alpha_2 = unidrnd(size(DS.alpha_2_vals,1));
    i_nc = unidrnd(size(DS.nc_vals,1)); 
    i_pb = unidrnd(size(problem_choices,1));
    % number of discretization point in each dimension 2^nc+1
    nc = DS.nc_vals(i_nc);
    % problem choice
    pb_choice = problem_choices(i_pb);
    % regularization parameter of the L2 norm
    alpha_2 = DS.alpha_2_vals(i_alpha_2);
    % regularization parameter of the L1 norm
    alpha_1 = DS.alpha_1_vals(i_alpha_1); 
    % control is constrained to be within range [u_alpha,u_beta]
    u_alpha = -2;                               
    u_beta = 1.5;
    if (pb_choice == 2)
        % diffusion coefficient
        epsilon = 0.05;    
    end
    % __________________________________________________________________________________________________ %
    % To accomodate all problem data.
    pb_struct = struct();
    % How large is resulting stiffness or mass matrix?
    np = (2^nc+1)^2; % entire system is thus 3*np-by-3*np
    % Compute matrices specifying location of nodes
    [x_1,x_2,x_1x_2,bound,mv,mbound] = square_domain_x(nc,grid_type);
    if (pb_choice == 1) % Poisson optimal control
        O = sparse(np,np);
        % Compute connectivity, stiffness and mass matrices (D and J)
        [ev,ebound] = q1grid(x_1x_2,mv,bound,mbound);
        [D,J_y] = femq1_diff(x_1x_2,ev);
        R = (sum(J_y))'; % equivalently diag(sum(J)), as J is symmetric
        R(bound) = 0;  % account for the boundary conditions
        
        % Specify vectors relating to desired state, source term and Dirichlet BCs
        yhat_vec = sin(pi*x_1x_2(:,1)).*sin(pi*x_1x_2(:,2));
        bc_nodes = ones(length(bound),1);

        % Initialize RHS vector corresponding to desired state
        Jyhat = J_y*yhat_vec;

        % Enforce Dirichlet BCs on state, and zero Dirichlet BCs on adjoint
        [D,b] = nonzerobc_input(D,zeros(np,1),x_1x_2,bound,bc_nodes);
        [J_y,Jyhat] = nonzerobc_input(J_y,Jyhat,x_1x_2,bound,bc_nodes);
        J_constr = J_y; for i = 1:length(bound), J_constr(bound(i),bound(i)) = 0; end
    elseif (pb_choice == 2) % Convection-diffusion optimal control
        O = sparse(np,np);
        % Compute connectivity, stiffness and mass matrices (D and J)
        [ev,ebound] = q1grid(x_1x_2,mv,bound,mbound);
        [K,N,J_y,epe,eph,epw] = femq1_cd(x_1x_2,ev);
        
        % Compute SUPG stabilization matrix
        epe = epe/epsilon;
        esupg = find(epe <= 1); expe = epe;
        if any(expe)
           supg = inf;
           if isinf(supg)
              expe = 0.5*(1-1./expe);
              expe(esupg) = inf;
           else
              expe = ones(size(expe)); expe = supg*expe; expe(esupg) = inf;
           end
           epp = expe; epp(esupg) = 0; epp = epp.*eph./epw;
           S = femq1_cd_supg(x_1x_2,ev,expe,eph,epw);
        end

        % Compute relevant matrices for optimization algorithm
        D = epsilon*K+N+S;
        
        R = (sum(J_y))'; % equivalently diag(sum(J)), as J is symmetric
        R(bound) = 0;  % account for the boundary conditions

        % Specify vectors relating to desired state and Dirichlet BCs
        yhat_vec = exp(-64.*((x_1x_2(:,1)-0.5).^2 + (x_1x_2(:,2)-0.5).^2));
        bc_nodes = 0.*x_1x_2(bound,1);
 
        % Initialize RHS vector corresponding to desired state
        Jyhat = J_y*yhat_vec;

        % Enforce Dirichlet BCs on state, and zero Dirichlet BCs on adjoint
        [D,b] = nonzerobc_input(D,zeros(np,1),x_1x_2,bound,bc_nodes);
        [J_y,Jyhat] = nonzerobc_input(J_y,Jyhat,x_1x_2,bound,bc_nodes);
        J_constr = J_y; for i = 1:length(bound), J_constr(bound(i),bound(i)) = 0; end
    else
        return;
    end 
    obj_const_term = (1/2).*(yhat_vec'*Jyhat);      % Constant term included in the objective.
    c = [-Jyhat; zeros(np,1)]; 
    J_u = alpha_2.*J_y;
    A = [D -J_constr];
    Q = [J_y     O ;
         O      J_u];
    L1_D = [zeros(np,1);alpha_1.*R];
    lb = -Inf.*ones(2*np,1);
    ub = Inf.*ones(2*np,1);
    lb(np+1:2*np) = u_alpha.*ones(np,1);
    ub(np+1:2*np) = u_beta.*ones(np,1);
    [A,L1_D,Q,c,b,lb,ub] = Problem_scaling_set_up(A,L1_D,Q,c,b,lb,ub,2*np,np);
    % Set the data struct that will contain all relevant problem info.
    pb_struct.A = A;    pb_struct.L1_D = L1_D;  pb_struct.Q = Q;
    pb_struct.c = c;    pb_struct.b = b;        pb_struct.lb = lb;
    pb_struct.ub = ub;  pb_struct.n = 2*np;     pb_struct.m  = np;
    pb_struct.ob_const = obj_const_term;        pb_struct.A_tr = A';
                                  
end

