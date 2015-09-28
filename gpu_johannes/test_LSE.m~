clear all

delete diary.txt
diary('diary.txt')
magicCUDA('solve_LSE_LU')

% N = 3;
N = 7000;
A = randn(N,N) + 50*eye(N); 
B = randn(N, 1);

tic
 W2 = A\B;
toc

%  cula_init();
 tic
 solve_LSE_LU(A, B);
 toc
%  cula_shutdown();
 
spy(abs(W2-B) > 1e-8)
