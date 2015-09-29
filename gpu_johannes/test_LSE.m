clear all

magicCUDA('solve_LSE_CULA_float')

N = 6890;
A = single(randn(N,N) + 50*eye(N)); 
B = single(randn(N, 30));
% tic
%  W2 = A\B;
% toc
for i=1:10,
    A = single(randn(N,N) + 50*eye(N)); 
    B = single(randn(N, 30));

    W2 = single(A\B);

    tic
    solve_LSE_CULA_float(A, B);
    toc
    
    spy(abs(W2-B) > 1e-4)
    pause;
end

% spy(abs(W2-B) > 1e-5)
