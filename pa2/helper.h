#ifndef __HELPER__
#define __HELPER__
#include <bits/stdc++.h>
using namespace std;
#define __HD__ __host__ __device__

#define SQR(x) ((x) * (x))

__inline__ __HD__ int rnd(int x, int n)
{
    return ((x + n - 1) / n) * n;
}
#define LG_BLOCK_N 64
#define LG_BLOCK_M 64
#define LG_BLOCK_K 8
#define LG_THREAD_N 8
#define LG_THREAD_M 4
#define LG_GRID_SIZE ((LG_BLOCK_N * LG_BLOCK_M) / (LG_THREAD_N * LG_THREAD_M))

#define SM_BLOCK_N 32
#define SM_BLOCK_M 32
#define SM_BLOCK_K 4
#define SM_THREAD_N 4
#define SM_THREAD_M 2
#define SM_GRID_SIZE ((SM_BLOCK_N * SM_BLOCK_M) / (SM_THREAD_N * SM_THREAD_M))


#endif
