%%
clear all
close all
clc

%magicCUDA('test')

k = 100;

M = load('./M_15000.mat');
M = M.M2;
N = load('./N_15000.mat');
N = N.M1;

M.evecs = M.evecs(:,1:k);
M.evals = M.evals(1:k);

N = N.M;
N.evecs = N.evecs(:,1:k);
N.evals = N.evals(1:k);

load('C_centaur.mat')

[C_refined, cpd_MN, R] = run_cpd(M, N, C_icp, 50);
cpd_MN = cpd_MN';

% if everything is correct, this should print 28.20
fprintf('Final accuracy: %.2f\n', 100*sum(M.gt==N.gt(cpd_MN))/M.n);

