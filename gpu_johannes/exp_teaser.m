
%%
clear all
close all
clc

% magicCUDA('solve_LSE_QR_GPU')
% magicCUDA('solve_LSE_LU')
% magicCUDA('solve_LSE_CULA_iterative_host')
% magicCUDA('solve_LSE_CULA_float_host')
% magicCUDA('solve_LSE_CULA_double_host')
% magicCUDA('cpd_P_GPU')
k = 100;
% k = 30;
% load('./tr_reg_079.mat');
% N = load('./tr_reg_084.mat');

load('./M_5000.mat');
N = load('./N_5000.mat');
M = M1;
N = N.M2;

M.evecs = M.evecs(:,1:k);
M.evals = M.evals(1:k);

% N = N.M;
N.evecs = N.evecs(:,1:k);
N.evals = N.evals(1:k);

% load('C_icp.mat')
load('C_wolf.mat')
 
[C_refined, cpd_MN, R] = run_cpd(M, N, C, 50);
cpd_MN = cpd_MN';

% if everything is correct, this should print 28.20
fprintf('Final accuracy: %.2f\n', 100*sum(M.gt==N.gt(cpd_MN))/length(M.gt));

