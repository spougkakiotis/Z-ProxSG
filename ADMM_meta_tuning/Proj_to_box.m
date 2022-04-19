function [y] = Proj_to_box(sigma_max,sigma_min,x,beta)
% This function simply projects x onto the box [sigma_min,sigma_max].    
    if (x < sigma_min)
        y = sigma_min;
    elseif (x > sigma_max)
        y = sigma_max;
    else
        y = x;
    end
end

