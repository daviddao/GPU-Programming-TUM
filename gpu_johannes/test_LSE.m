clear all

delete diary.txt
diary('diary.txt')
magicCUDA('solve_LSE_CULA')

% N = 3;
N = 8000;
A = randn(N,N) + 50*eye(N); 
B = randn(N, 30);
X = zeros(N, 30);
tic
 W2 = A\B;
toc

 tic
 solve_LSE_CULA(A, B, X);
 toc
 
%  tic
%  [L, U, P] = lu(A);
%  X1 = L\B;
%  X2 = U\X1;
%  toc

spy(abs(W2-X) > 1e-8)
