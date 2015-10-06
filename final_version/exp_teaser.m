%%
clear all
close all
clc

k = 30;

%magicCUDA('cpd_P_GPU');
%magicCUDA('solve_LSE_CULA_float');


load('./tr_reg_079.mat');
N = load('./tr_reg_084.mat');

M.evecs = M.evecs(:,1:k);
M.evals = M.evals(1:k);

N = N.M;
N.evecs = N.evecs(:,1:k);
N.evals = N.evals(1:k);

load('C_icp.mat')

[C_refined, cpd_MN, R] = run_cpd(M, N, C_icp, 50);
cpd_MN = cpd_MN';

% if everything is correct, this should print 28.20
fprintf('Final accuracy: %.2f\n', 100*sum(M.gt==N.gt(cpd_MN))/M.n);
