function magicCPP( string )
%MAGIC Summary of this function goes here
% Detailed explanation goes here
% magic cuda compilation

newstring = strcat(string,'.cpp');
ostring = strcat(string,'.o');
commandstr = ['g++ -O3 -fopenmp -DNDEBUG -c ' newstring ' -fPIC -I/usr/include/eigen3 -I/usr/local/lehrstuhl/DIR/matlab-R2015a/extern/include -I/usr/local/lehrstuhl/DIR/matlab-R2015a/toolbox/distcomp/gpu/extern/include']

unix(commandstr)
commandstr2 = ['mex ', ostring, ' -L/usr/local/lehrstuhl/DIR/matlab-R2015a/bin/glnxa64 -lmwgpu']
eval(commandstr2)

end 

