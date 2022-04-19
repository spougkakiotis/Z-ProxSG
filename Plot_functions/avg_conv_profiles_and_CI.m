function [] = avg_conv_profiles_and_CI(fig,x,y_mat,color,dot_type,line_type,x_label,y_label)
% ================================================================================== %
% Plot average convergence profiles and CI intervals
% ---------------------------------------------------------------------------------- %
% ** [] = avg_conv_profiles_and_CI(x,y_mat) **
%
% This function plots the average convergence of experiment runs of an algorithms
% as well as their 95% confidence interval.
%
% INPUT (ordered):
%   • fig       - figure upon which to plot
%   • x         - the x-axis along which we plot the data (i.e. iterations)
%   • y_mat     - a matrix containing certain runs of an algorithm over  
%                 max(size(x,1), size(x,2)) iterations
%   • color     - string indicating the color of the line
%   • dot_type  - string indicating the type of dots in the plot
%   • line_type - string indicating the line type of the average plot
%   • x_label   - the label of the x-axis.
%   • y_label   - the label of the y-axis.
% OUTPUT: Plots the result
%       
% Author: Star Strider, Mathworks
% __________________________________________________________________________________ %
    if (nargin < 4 || isempty(color))
        color = 'b';
    end
    if (nargin < 5 || isempty(dot_type))
        dot_type = 'o';
    end
    if (nargin < 6 || isempty(line_type))
        line_type = '-';
    end
    plot_str = strcat(color,dot_type,line_type);


    if (size(x,1) > 1)
        x = x';
    end
    if (size(y_mat,1) == size(x,2))
        y_mat = y_mat';
    end
    if (size(y_mat,2) ~= size(x,2))
        error('Incorrect input dimensions.\n');
    end
    N = size(y_mat,1);                                    % Number of ‘Experiments’ In Data Set
    yMean = mean(y_mat);                                  % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = (std(y_mat)/sqrt(N));                          % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                      % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));                % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
    figure(fig)
    grid off
    xlabel(x_label) 
    ylabel(y_label) 
    plot(x, yMean,plot_str,'LineWidth',3)                 % Plot Mean Of All Experiments
    hold on
    % Plot 95% Confidence Intervals Of All Experiments
    patch([x, fliplr(x)], [yMean + yCI95(1,:) fliplr(yMean+yCI95(2,:))], color, 'EdgeColor',... 
          'none', 'FaceAlpha',0.25)
    %hold off
    grid
end

