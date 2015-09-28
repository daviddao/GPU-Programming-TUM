/**
 * David Dao, Johannes Rausch, Michal Szymczak
 * TU Munich
 * Sep 2015
 */

#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cula_device.h>
#include <cula_lapack_device.h>

using namespace std;

void checkStatus(culaStatus status)
{
char buf[80];

if(!status)
return;

culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
printf("%s\n %d", buf, status);

culaShutdown();
//exit(EXIT_FAILURE);
}


/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  int device_pointer = 34;
  char buf[256];
  culaStatus status = culaInitialize();




  checkStatus(status);
  checkStatus(culaGetExecutingDevice(&device_pointer));
    printf("Device: %d\n", device_pointer);
  checkStatus(culaGetDeviceInfo(device_pointer, buf, 256));
  for(int i=0;i<256;i++)
    printf("%c", buf[i]);
    printf("\n");

return;
}



