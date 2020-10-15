#include<stdio.h>
#include<math.h>
#include "sigmoid.h"

void sigmoidForward(float *x, float *y, int dataLen)
{
    for (int i=0; i<dataLen; i++)
        *(y + i) = 1.0 / (1.0 + expf(-*(x + i)));
}

void sigmoidBackward(float *deltaLoss, float *Loss, float *y, int dataLen)
{
    for (int i=0; i<dataLen; i++)
        *(deltaLoss + i) = *(Loss + i) * (1.0 - *(y + i)) * *(y + i);
}