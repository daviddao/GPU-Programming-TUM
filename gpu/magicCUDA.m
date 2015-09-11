function magicCUDA( string )
%MAGIC Summary of this function goes here
% Detailed explanation goes here
% magic cuda compilation

newstring = strcat(string,'.cu') 
ostring = strcat(string,'.o') 
commandstr = ['nvcc -O3 -DNDEBUG -c ' newstring ' -Xcompiler -fPIC -I/usr/local/lehrstuhl/DIR/matlab-R2015a/extern/include -I/usr/local/lehrstuhl/DIR/matlab-R2015a/toolbox/distcomp/gpu/extern/include']


unix(commandstr)
commandstr2 = ['mex ', ostring, ' -L/usr/local/cuda-7.0/lib64 -L/usr/local/lehrstuhl/DIR/matlab-R2015a/bin/glnxa64 -lcudart -lcufft -lmwgpu']
eval(commandstr2)


end

