function [flowx,flowy] = specific_wind(x,y,nel)
%SPECIFIC_WIND (current) problem convective wind 
%   [flowx,flowy] = specific_wind(x,y,nel);
%   input
%          x          x coordinate vector
%          y          y coordinate vector 
%          nel        number of elements
%
%   specifies circular wind /Morton pp.10/
%   IFISS function: DJS; 5 March 2005.
% Copyright (c) 2005 D.J. Silvester, H.C. Elman, A. Ramage 
      flowx =  2*y.*(1-x.*x);       
      flowy = -2*x.*(1-y.*y);
      return
