
delete diary.txt
diary('diary.txt')
magicCUDA('solve_LSE_GPU')

A = [1 2 3; 2 1 -3; 0 9 1];
B = [1 1 3];

solve_LSE_GPU(A, B)