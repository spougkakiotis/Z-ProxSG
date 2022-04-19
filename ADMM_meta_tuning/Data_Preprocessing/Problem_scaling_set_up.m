function [A,D,Q,c,b,lb,ub] = Problem_scaling_set_up(A,D,Q,c,b,lb,ub,n,m)

    % ==================================================================================================================== %
    % Scale the problem matrix
    % -------------------------------------------------------------------------------------------------------------------- %
    scaling_direction = 'l';
    scaling_mode = 2;
     if (scaling_direction == 'r') % Apply the right scaling.
        [D_sc] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = A*spdiags(D_sc,0,n,n); 
        c = c.*D_sc;
        Q = spdiags(D_sc,0,n,n)*Q*spdiags(D_sc,0,n,n);
        D = D_sc.*D;
        lb = lb./D_sc;
        ub = ub./D_sc;
    elseif (scaling_direction == 'l') % Apply the left scaling.
        [D_sc] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = spdiags(D_sc,0,m,m)*A;                         
        b = b.*D_sc;
     elseif (scaling_direction == 'b') % Apply left and right scaling
        [D_scR] = Scale_the_problem(A,scaling_mode,'r');
        [D_scL] = Scale_the_problem(A,scaling_mode,'l');
        A = spdiags(D_scL,0,m,m)*A*spdiags(D_scR,0,n,n);
        b = b.*D_scL;
        c = c.*D_scR;
        D = D_scR.*D;
        Q = spdiags(D_scR,0,n,n)*Q*spdiags(D_scR,0,n,n);
        lb = lb./D_scR;
        ub = ub./D_scR;
     end
    % ____________________________________________________________________________________________________________________ %
end

