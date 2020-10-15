#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include<stdio.h>
#include<math.h>

extern void sigmoidForward(float *x, float *y, int len);
extern void sigmoidBackward(float *deltaLoss, float *Loss, float *y, int len);

#endif // __SIGMOID_H__