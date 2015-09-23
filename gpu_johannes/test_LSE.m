clear all

delete diary.txt
diary('diary.txt')
magicCUDA('solve_LSE_CULA')

% N = 3;
N = 1500;
A = randn(N,N); 
B = [1; 1; 3];

A1 = A;

% tic
% [L, U, P] = lu(A);
% toc


tic
solve_LSE_CULA(A)
toc
L1 = tril(A,-1) + eye(size(A));
U1 = triu(A);

spy(abs(A1-L1*U1) > 10e-3)

