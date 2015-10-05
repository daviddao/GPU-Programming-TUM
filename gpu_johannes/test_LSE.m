clear all

% magicCUDA('solve_LSE_CULA_double_host')

N = 6890;
M = 30;
% A = (randn(N,N)); 
% B = (randn(N, 30));
% X = zeros(N,30);

for i=1:4,
    A = single((randn(N,N))); 
    B = single((randn(N, M)));
    X = single(zeros(N,M));
    
    tic
    W2 = A\B;
    toc 
     
    tic
    X = solve_LSE_QR_GPU(A, B);
    toc
    
    spy(abs(W2-X) > 1e-3)
%     pause;
end

% spy(abs(W2-B) > 1e-5)
