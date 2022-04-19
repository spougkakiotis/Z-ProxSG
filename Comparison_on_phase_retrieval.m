% ================================================================================== %
% Phase retrieval problem
% ---------------------------------------------------------------------------------- %
% The optimization problem has the following form:
%
% \[            min_{x \in R^d} f(x) = (1/m) \sum_{i = 1}^{m} f_i(x),             \]
% 
% where $f_i(x) = |(a_i^T x)^2-b_i|$, for $i = 1,...,m$, $x \in R^d$,  
% where the data are randomly generated. 
%
% Experimental setups (same as in Davis, Drusvyatskiy, "Stochastic model-base
% minimization of weakly convex functions").
%
% • (d,m) = (10,30), (50,150), (100,300)
%
% Author: Spyridon Pougkakiotis, April 2022, Connecticut.
% _________________________________________________________________________________ %
    clear;
    clc;
    % Include (recursively) all subfolders.
    curr_path = pwd;
    addpath(genpath(curr_path)); 
    % ============================================================================= %
    % Problem generation
    % ----------------------------------------------------------------------------- %
    % Set the rng seed.
    rng('shuffle');
    % Problem dimensions (d = "size of unknown variable", m = "number of samples").
    d = 30; m = 90;
    % number of iterations to termination (as in Davis et. al.)
    T = 2e3*m;
    % Number of runs that we average
    averaging = 15;
    % Plot convergence profiles?
    plot_f = true;
    % Which experiment to run? 1 for comparison, 2 for minibatching
    experiment = 2; 
    if (plot_f && experiment == 1)
        % objective function value within each run
        obj_all_Z_proxSG = zeros(T+1,averaging);
        obj_all_TPZ_proxSG = zeros(T+1,averaging);
        obj_all_SPMM = zeros(T+1,averaging);
        obj_all_proxSSG = zeros(T+1,averaging);
    elseif (plot_f && experiment == 2)
        obj_all_Z_proxSG_k_1 = zeros(T+1,averaging);
        obj_all_Z_proxSG_k_4 = zeros(T+1,averaging);
        obj_all_Z_proxSG_k_8 = zeros(T+1,averaging);        
    end
    if (plot_f)
        fig = figure;
    end
    for j = 1:averaging
        % Generate the problem data, as well as function evaluations of $f()$,
        % subgradient evaluations of $f_i()$ and prox evaluations w.r.t. $f_i()$.
        % For more, type "help phase_retrieval_examples"
        [pb_struct] = phase_retrieval_examples(d,m);
        % _________________________________________________________________________ %
        if (experiment == 1)
            % ===================================================================== %
            % Call Z_proxSG to solve the problem (Zeroth-order stochastic gradient)
            % --------------------------------------------------------------------- %
            % Smoothing parameter
            mu = 5e-10;
            % Learning rate
            alpha = (1/(sqrt(min(T,5e2*m))*(2*d)));
            % number of gradient estimates per iteration
            k = 1;

            % Call Z_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = Z_proxSG(@(y,i) pb_struct.f_i(y,i),@(y,beta) y,...
                                    @() pb_struct.sample_xi(),mu,...
                                    pb_struct.data.start_point,T,alpha,k,...
                                    plot_f, @(y) pb_struct.f(y));
            if (plot_f)
                obj_all_Z_proxSG(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %


            % ===================================================================== %
            % Call TPZ_proxSG to solve the problem (Zeroth-order stochastic gradient)
            % --------------------------------------------------------------------- %
            % Smoothing parameter
            mu_1 = 5e-10;
            mu_2 = 2.5e-10;
            % Learning rate
            alpha = (1/(sqrt(min(T,5e2*m))*(2*d)));
            % number of gradient estimates per iteration
            k = 1;

            % Call TPZ_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = TPZ_proxSG(@(y,i) pb_struct.f_i(y,i),@(y,beta) y,...
                                    @() pb_struct.sample_xi(),mu_1,mu_2,...
                                    pb_struct.data.start_point,T,alpha,k,...
                                    plot_f, @(y) pb_struct.f(y));
            if (plot_f)
                obj_all_TPZ_proxSG(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %
            % ===================================================================== %
            % Call Stoch_PPM to solve the problem (Stochastic proximal point method)
            % --------------------------------------------------------------------- %
            % Learning rate
            alpha = 1.1e-1;
            % Call Z_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = Stoch_PPM(@(beta,y,i) pb_struct.prox_f_i(beta,y,i), ...
                                     @() pb_struct.sample_xi(), ...
                                     pb_struct.data.start_point,T,alpha, ...
                                     plot_f,@(y) pb_struct.f(y));                            
            if (plot_f)
                obj_all_SPMM(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %

            % ===================================================================== %
            % Call ProxSSG to solve the problem (Stochastic sub-gradient)
            % --------------------------------------------------------------------- %
            % Learning rate
            alpha = 1/(2*sqrt(min(T,5e2*m)));
            % Call Z_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = ProxSSG(@(y,i) pb_struct.sub_f_i(y,i), ...
                                   @(y,beta) y, @() pb_struct.sample_xi(), ...
                                   pb_struct.data.start_point,T,alpha,...
                                   plot_f, @(y) pb_struct.f(y));                            
            if (plot_f)
                obj_all_proxSSG(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %
        elseif (experiment == 2)
            % Smoothing parameter
            mu = 5e-10;
            % Learning rate
            alpha = (1/(sqrt(min(T,5e2*m))*(2*d)));
            % ===================================================================== %
            % Call Z_proxSG with k=1
            % --------------------------------------------------------------------- %
            % number of gradient estimates per iteration
            k = 1;

            % Call Z_proxSG
            [sol_struct] = Z_proxSG(@(y,i) pb_struct.f_i(y,i),@(y,beta) y,...
                                    @() pb_struct.sample_xi(),mu,...
                                    pb_struct.data.start_point,T,alpha,k,...
                                    plot_f, @(y) pb_struct.f(y));
            if (plot_f)
                obj_all_Z_proxSG_k_1(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %


            % ===================================================================== %
            % Call Z_proxSG with k=2
            % --------------------------------------------------------------------- %
            % number of gradient estimates per iteration
            k = 4;

            % Call Z_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = Z_proxSG(@(y,i) pb_struct.f_i(y,i),@(y,beta) y,...
                                    @() pb_struct.sample_xi(),mu,...
                                    pb_struct.data.start_point,T,alpha,k,...
                                    plot_f, @(y) pb_struct.f(y));
            if (plot_f)
                obj_all_Z_proxSG_k_4(:,j) = sol_struct.obj_v;
            end
            % _____________________________________________________________________ %


            % ===================================================================== %
            % Call Z_proxSG with k=4
            % --------------------------------------------------------------------- %
            % number of gradient estimates per iteration
            k = 8;

            % Call Z_proxSG; the proximity operator of r() is the identity function
            [sol_struct] = Z_proxSG(@(y,i) pb_struct.f_i(y,i),@(y,beta) y,...
                                    @() pb_struct.sample_xi(),mu,...
                                    pb_struct.data.start_point,T,alpha,k,...
                                    plot_f, @(y) pb_struct.f(y));
            if (plot_f)
                obj_all_Z_proxSG_k_8(:,j) = sol_struct.obj_v;
            end
            % _________________________________________________________________________ %
        else
            return
        end
    end
    x_axis = (0:T)';
    xlabel('T') 
    ylabel('f(x)')
    if (experiment == 1)
        if (plot_f)
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_Z_proxSG,[],[],[],'T','f(x)');
            hold on;
            grid off
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_TPZ_proxSG,'c','x',':','T','f(x)');
            hold on;
            grid off    
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_SPMM,'g','x','--','T','f(x)');
            hold on;
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_proxSSG,'r','+','-.','T','f(x)');
            legend("ZproxSG: Average","                95% CI","TPZproxSG: Average","                     95% CI",...
                   "StochPPM: Average","                   95% CI",...
                   "ProxSSG: Average","                 95% CI");
            hold off;
        end   
    elseif (experiment == 2)
        if (plot_f)
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_Z_proxSG_k_1,[],[],[],'T','f(x)');
            hold on;
            grid off
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_Z_proxSG_k_4,'c','x',':','T','f(x)');
            hold on;
            grid off
            avg_conv_profiles_and_CI(fig,x_axis,obj_all_Z_proxSG_k_8,'g','x','--','T','f(x)');
            legend("ZproxSG (k=1): Average","                          95% CI",...
                   "ZproxSG (k=4): Average","                          95% CI",...
                   "ZproxSG (k=8): Average","                          95% CI");
            hold off;
        end           
    end



