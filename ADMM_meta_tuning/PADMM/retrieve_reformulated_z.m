function [z] = retrieve_reformulated_z(D,w,z)
% ================================================================================ %
% Return reformulated dual multipliers used within SSN-PMM.
% -------------------------------------------------------------------------------- %
% Letting g(x) = \|Dx\|_1, but assuming D is diagonal (hence given as a vector),
% compute z of the original reformulation as:
%                   z = z - Proj_{\partial g(w)}(z)
% ________________________________________________________________________________ %

    tmp = z;
    I_1 = find(w > 0);      % Sets that characterize the subdifferential of g(w).
    I_2 = find(w < 0);
    I_3 = find(w == 0);
    tmp(I_1) = D(I_1);      % Projections to singleton sets.
    tmp(I_2) = -D(I_2);
    I_4 = find(tmp(I_3) < -D(I_3));
    I_5 = find(tmp(I_3) > D(I_3));
    tmp(I_3(I_4)) = -D(I_3(I_4));
    tmp(I_3(I_5)) = D(I_3(I_5));
    z = z - tmp;
end

