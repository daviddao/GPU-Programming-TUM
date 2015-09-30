function magicCUDA( string )
%MAGIC Summary of this function goes here
% Detailed explanation goes here
% magic cuda compilation

newstring = strcat(string,'.cu');
ostring = strcat(string,'.o');
commandstr = ['nvcc -O3 -DNDEBUG -c ' newstring ' --gpu-architecture=sm_20 -lcula_lapack -lcublas -Xcompiler -fPIC -I$CULA_INC_PATH -I/usr/local/lehrstuhl/DIR/matlab-R2015a/extern/include -I/usr/local/lehrstuhl/DIR/matlab-R2015a/toolbox/distcomp/gpu/extern/include']
unix(commandstr)

commandstr2 = ['mex ', ostring, ' -L/usr/local/cuda/lib64 -L$CULA_LIB_PATH_64 -L/usr/local/lehrstuhl/DIR/matlab-R2015a/bin/glnxa64 -lcula_lapack -lcusolver -lcublas -lcusparse -lcudart -lcufft -lmwgpu']
eval(commandstr2)


end
