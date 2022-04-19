% ================================================================================== %
% ADMM Meta-tuning
% ---------------------------------------------------------------------------------- %
% In this script, we employ a preconditioned ADMM for the solution of 
% Poisson optimal control problems of the following form
%
% \[              min_{x \in R^d} (1/2)x^T Q x + c^T x + \|Dx\|_1,      
%                 s.t.            Ax = b, x \in [lb,ub]
%                                                                                 \]
% 
% where the constraints arise as the discretization of the Poisson equation,
% and the bound constraints restrict the control variable.
%
% We employ various discretized instances of this problem with different 
% parameters as a "training set", and repeatedly apply a few iterations
% of the ProxADMM to assess the quality of the proximal parameter sigma,
% and tune it using a zeroth order stochastic proximal gradient method.
% Thus we employ a gradient-free scheme for tuning the ADMM penalty parameter.
%
% Experimental setup:
%
% We discretize the Poisson optimal control problem using the Q1 finite-element 
% discretization and choose 4 different problem sizes, 5 different values for 
% the L^2 regularization parameter and 5 different values for the L^1 
% regularization of the model. This yields 100 different optimization instances
% that act as a training set for tuning the ProxADMM for similar PDE-constrained
% optimization instances.
%
% The quality of the penalty parameter of the ProxADMM is measured by the residual
% reduction each run of ProxADMM achieves after 15 iterations. The smaller 
% the reduction, the better the quality. Thus, we employ a zeroth-order 
% scheme for the solution of the following optimization problem
%
%       min_{sigma}   f(sigma) = E[F(sigma,xi)], where xi \sim Xi
%       s.t.          sigma \in [1e-2,1e2],
%
% where the bounds on sigma reflect that very small or very large values for this 
% penalty are known to behave badly and hence need not be considered, while 
%
%      F(sigma,Xi) = res_reduction_of_ProxADMM(sigma,maxit=15,xi),
%
% and xi is a uniformly-chosen random number from 1 to 100, randomly picking one of  
% the 100 available instances. This is the training set, which we expect to 
% generalize the behaviour of ProxADMM on other similar PDE-constrained 
% optimization instances.
%
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% __________________________________________________________________________________ %



    % Include Code and Data files.
    curr_path = pwd;
    addpath(genpath(curr_path)); 
    fid = 1;   
    training = false;
    if (training)
        % Tolerance used for all experiments.
        tol = 1e-6;                                             
        % Maximum number of ADMM iterations.
        max_ADMM_iter = 15;  
    end
    % Initial ADMM penalty parameter.
    sigma_0 = 1;                                            
    % A struct containing all relevant data defining a problem instance.
    DS = struct();   
    % Possible parameter values that define 100 optimization instances.
    DS.alpha_1_vals = [0; 1e-2; 1e-4; 1e-6;];
    DS.alpha_2_vals = [0; 1e-2; 1e-4; 1e-6;];
    DS.nc_vals      = [3; 4; 5; 6; 7];
    % Acceptable range for the penalty parameter.
    sigma_max       = 1e2;
    sigma_min       = 1e-2;
    % ============================================================================== %
    % Problem choice: we solve Poisson optimal control problems
    % problem_choice = 1 -> Poisson optimal control,
    % problem_choice = 2 -> Convection-diffusion optimal control.
    % ------------------------------------------------------------------------------ %
    problem_choice = 1;
    % ______________________________________________________________________________ %
    % Set the rng seed.
    rng('shuffle');
    if (training)
        % sample and variable size
        m = size(DS.alpha_1_vals,1)*size(DS.alpha_2_vals,1)*size(DS.nc_vals,1);
        d = 1;          % only sigma is the unknown parameter.
        % number of iterations to termination of the zeroth-order method.
        T = 5e2*m;

        % ========================================================================= %
        % Call Z_proxSG to solve the problem (Zeroth-order stochastic gradient)
        % ------------------------------------------------------------------------- %
        % Smoothing parameter
        mu = 5e-10;
        % Learning rate
        alpha = (1/(sqrt(min(T,1e2*m))*(2*d)));
        % number of gradient estimates per iteration
        k = 1;
        sample_problem = @() Pearson_PDE_Test_Generator(problem_choice,DS);
        Obj_fun_sample = @(x,sample) ProxADMM(x,sample,tol,max_ADMM_iter); 
        prox_r = @(x,beta) Proj_to_box(sigma_max,sigma_min,x,beta);

        % Call Z_proxSG; the proximity operator of r() is the identity function
        [sol_struct] = Z_proxSG(@(x,sample) Obj_fun_sample(x,sample),...
                                @(x,beta) prox_r(x,beta), @() sample_problem(),...
                                mu,sigma_0,T,alpha,k);
        % _________________________________________________________________________ %
    else
        % Plot convergence profiles?
        plot_R = true;

        if (plot_R)
            fig = figure;
        end

        % Tolerance used for all experiments.
        tol = 1e-6;           
        if (problem_choice == 1)
             % Maximum number of ADMM iterations.
            max_ADMM_iter = 400;
            % Optimal solution found from Z-ProxSG (Poisson problem)
            sigma_opt = 0.2778;
            % Equispaced values of sigma in [10^(-2),10^2]
            sigma_vals = [sigma_opt; 5e-2; 0.8; 2; 5; 50]; % Poisson example
        else % Convection-diffusion example
            max_ADMM_iter = 800; 
            % Optimal solution found from Z-ProxSG (Convection-diffusion problem)
            sigma_opt = 5.7004;
            sigma_vals = [sigma_opt; 5e-2; 1e-1; 5e-1; 20; 50]; 
        end
        % Averaging
        averaging = 40;
        OS_DS = struct();   
        % Possible parameter values that define 96 out-of-sample instances.
        OS_DS.alpha_1_vals = [1e-3; 5e-3; 1e-5; 5e-5];
        OS_DS.alpha_2_vals = [1e-3; 5e-3; 1e-5; 5e-5];
        OS_DS.nc_vals      = [3; 4; 5; 6; 7; 8;];

        res_all_Z_proxSG = zeros(size(sigma_vals,1)*max_ADMM_iter,averaging);
        for i = 1:averaging
            % Sample a problem
            pb_struct = Pearson_PDE_Test_Generator(problem_choice,OS_DS);
            for j = 1:(size(sigma_vals,1))
                % For each value of sigma, call ProxADMM
                [~,ADMM_sol_struct] = ProxADMM(sigma_vals(j),pb_struct,tol,max_ADMM_iter,plot_R);
                if (plot_R)
                    % If plot is activated, store all intermediate values of the scaled residual norm.
                    res_all_Z_proxSG((j-1)*max_ADMM_iter+1:j*max_ADMM_iter,i) = ADMM_sol_struct.res_norm; 
                end        
            end
        end

        x_axis = (1:max_ADMM_iter)';
        xlabel('Iteration') 
        ylabel('min(Scaled residual,0.5)')
        if (plot_R)
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(1:max_ADMM_iter,:),[],[],[],'Iteration','min(Scaled residual,0.5)');
            hold on;
            grid off
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(max_ADMM_iter+1:2*max_ADMM_iter,:),'c','x',':','Iteration','min(Scaled residual,0.5)');
            hold on;
            grid off    
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(2*max_ADMM_iter+1:3*max_ADMM_iter,:),'g','d','--','Iteration','min(Scaled residual,0.5)');
            hold on;
            grid off
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(3*max_ADMM_iter+1:4*max_ADMM_iter,:),'r','+','-.','Iteration','min(Scaled residual,0.5)');
            hold on;
            grid off  
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(4*max_ADMM_iter+1:5*max_ADMM_iter,:),'y','s','-','Iteration','min(Scaled residual,0.5)');
            hold on;
            grid off  
            avg_conv_profiles_and_CI(fig,x_axis,...
                   res_all_Z_proxSG(5*max_ADMM_iter+1:6*max_ADMM_iter,:),'k','h','-.','Iteration','min(Scaled residual,0.5)');

            legend("Average: sigma = 5.7004","95% CI","Average: sigma = 0.05","95% CI",...
                   "Average: sigma = 0.1","95% CI","Average: sigma = 0.5","95% CI",...
                   "Average: sigma = 20","95% CI","Average: sigma = 50","95% CI");
            hold off;
        end   
    end
