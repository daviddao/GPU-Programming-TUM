function [C_refined, matches, R] = run_cpd(M, N, C_init, max_iters)

options = struct;
options.outliers = 0.1;
options.beta = 2;   % width of Gaussian (band of low-pass)
options.lambda = 3; % regularization weight
options.max_it = max_iters;
options.tol = 1e-5;
options.sigma2 = 0;

N_pts = N.evecs;
M_pts = M.evecs*C_init';



% [~, matches] = cpd_register(N_pts, M_pts, options);
[~, matches] = cpd_register(N_pts, M_pts, options);

C_refinement = N_pts(matches,:)\M_pts;
C_refined = C_refinement' * C_init;

if nargout==3
    R = C_refinement';
end



end
