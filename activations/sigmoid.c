#include<stdio.h>
#include<math.h>

void sigmoidForward(float *x, float *y, int len)
{
    for (int i=0; i<len; i++)
        *(y + i) = 1.0 / (1.0 + expf(-*(x + i)));
}

void sigmoidBackward(float *deltaLossX, float *deltaLossY, float *y, int xLen)
{
}