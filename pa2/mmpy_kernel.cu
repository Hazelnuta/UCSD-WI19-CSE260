// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include <cuda.h>
#include <bits/stdc++.h>
using namespace std;

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
//
// __global__ void g_matMulo(int N, double *C, double *A, double *B)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//     double c = 0;
//     int x = idx / N, j = idx % N;
//
//     if (idx >= N * N) return;
//     for (int k = 0; k < N; k++)
//         c += A[x * N + k] * B[k * N + j];
//     C[idx] = c;
// }
//
// __global__ void printMat(int N, double *A)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//     if (idx >= N * N) return;
//     printf("%d %d %f\n", idx / N, idx % N, A[idx]);
// }
// #define BLOCK_SIZE_SHIFT 5
// #define BLOCK_SIZE (1 << BLOCK_SIZE_SHIFT)
// #define BLOCK_SIZE_MASK (BLOCK_SIZE - 1)
//
// #define BLOCK_PORTION_SHIFT 3
// #define BLOCK_PORTION (1 << BLOCK_PORTION_SHIFT)
// #define BLOCK_PORTION_MASK (BLOCK_PORTION - 1)
// __global__ void matMulB(int N, int blkN, double *C, const double *const __restrict__ A, const double *const __restrict__ B)
// {
//     __shared__ double sA[BLOCK_SIZE][BLOCK_SIZE], sB[BLOCK_SIZE][BLOCK_SIZE];
//     int blkx = blockIdx.x / blkN;
//     int blky = blockIdx.x % blkN;
//  // int thdx = threadIdx.x >> BLOCK_SIZE_SHIFT;
//     int thdx = threadIdx.x / BLOCK_SIZE;
//  // int thdy = threadIdx.x & BLOCK_SIZE_MASK;
//     int thdy = threadIdx.x % BLOCK_SIZE;
//  // int x = (blkx << BLOCK_SIZE_SHIFT) | thdx, y = (blky << BLOCK_SIZE_SHIFT) | thdy;
//     int x = blkx * BLOCK_SIZE + thdx, y = blky * BLOCK_SIZE + thdy;
//     double c[BLOCK_PORTION] = { 0 };
//     int idxA = x * N + thdy, idxB = thdx * N + y;
//
// #pragma unroll
//     for (int i = 0; i < blkN; i++) {
// #pragma unroll
//         for (int p = 0; p < BLOCK_PORTION; p++) {
//             sA[thdx + p * BLOCK_SIZE / BLOCK_PORTION][thdy] = A[idxA + N * p * BLOCK_SIZE / BLOCK_PORTION];
//             sB[thdx + p * BLOCK_SIZE / BLOCK_PORTION][thdy] = B[idxB + N * p * BLOCK_SIZE / BLOCK_PORTION];
//         }
//         __syncthreads();
//         idxA += BLOCK_SIZE;
//      // idxB += N << BLOCK_SIZE_SHIFT;
//         idxB += N * BLOCK_SIZE;
// #pragma unroll
//         for (int k = 0; k < BLOCK_SIZE; k++) {
//             double tmpb = sB[k][thdy];
// #pragma unroll
//             for (int p = 0; p < BLOCK_PORTION; p++)
//                 c[p] += sA[thdx + p * BLOCK_SIZE / BLOCK_PORTION][k] * tmpb;
//         }
//         __syncthreads();
//     }
// #pragma unroll
//     for (int p = 0; p < BLOCK_PORTION; p++)
//         C[(x + p * BLOCK_SIZE / BLOCK_PORTION) * N + y] = c[p];
// }
//
// #define BLOCK_R_SHIFT 6
// #define BLOCK_R (1 << BLOCK_R_SHIFT)
// #define BLOCK_C_SHIFT 5
// #define BLOCK_C (1 << BLOCK_C_SHIFT)
// __global__ void g_matMulVD(int N, double *C, const double *const __restrict__ A, const double *const __restrict__ B)
// {
//     __shared__ double sB[BLOCK_C][BLOCK_C];
//     int idxC, idxA, idxB;
//     double c[BLOCK_C] = { 0 };
//
//  // idxC = blockIdx.x * BLOCK_R * N + blockIdx.y * BLOCK_C;
//  // idxA = blockIdx.x * BLOCK_R * N;
//  // idxB = blockIdx.y * BLOCK_C;
//     idxB = blockIdx.y << BLOCK_C_SHIFT;
//     idxA = blockIdx.x * N << BLOCK_R_SHIFT;
//     idxC = idxA + idxB;
//     for (int i = 0; i < N >> BLOCK_C_SHIFT; i++) {
// #pragma unroll
//         for (int j = 0; j < (1 << (BLOCK_C_SHIFT + BLOCK_C_SHIFT - BLOCK_R_SHIFT)); j++)
//             sB[(threadIdx.x >> BLOCK_C_SHIFT) + (j << (BLOCK_R_SHIFT - BLOCK_C_SHIFT))][threadIdx.x & (BLOCK_C - 1)] = B[idxB + N * ((threadIdx.x >> BLOCK_C_SHIFT) + (j << (BLOCK_R_SHIFT - BLOCK_C_SHIFT))) + (threadIdx.x & (BLOCK_C - 1))];
//         __syncthreads();
//         idxB += N << BLOCK_C_SHIFT;
//         for (int j = 0; j < BLOCK_C; j++) {
//             double a = A[idxA + N * threadIdx.x + j];
// #pragma unroll
//             for (int k = 0; k < BLOCK_C; k++)
//                 c[k] += a * sB[j][k];
//         }
//         idxA += BLOCK_C;
//         __syncthreads();
//     }
// #pragma unroll
//     for (int i = 0; i < BLOCK_C; i++)
//         C[idxC + threadIdx.x * N + i] = c[i];
// }
// #define GS ((BN * BM) / (TN * TM))
// __shared__ double sA[64][8], sB[8][64];
template<int BN, int BM, int BK, int TN, int TM, int GS>
__device__ __inline__ void block_matMul(int N, double *__restrict__ C, const double *const __restrict__ A, const double *const __restrict__ B)
{
    __shared__ double sA[BN][BK], sB[BK][BM];
    int thdx, thdy;
    double c[TN][TM] = { 0 };
    double ra[TN], rb[TM];
    int x, y;
    int xA, xB, xC;
    int yA, yB, yC;

    xA = blockIdx.x * BN;
    xA = blockIdx.x * BN;
    yA = 0;
    xB = 0;
    yB = blockIdx.y * BM;
    thdx = threadIdx.x / (BM / TM);
    thdy = threadIdx.x % (BM / TM);
    xC = xA + xB + thdx * TN;
    yC = yA + yB + thdy * TM;
    for (int i = 0; i < N; i += BK) {
		// copy sA
        y = threadIdx.x % BK;
#pragma unroll
        for (int j = 0; j < (BN * BK + GS - 1) / GS; j++) {
            x = j * GS / BK + threadIdx.x / BK;
            if (xA + x < N)
                sA[x][y] = A[(xA + x) * N + yA + y];
            else sA[x][y] = 0;
        }
		// copy sB
        y = threadIdx.x % BM;
#pragma unroll
        for (int j = 0; j < (BM * BK + GS - 1) / GS; j++) {
            x = threadIdx.x / BM + j * GS / BM;
            if (xB + x < N)
                sB[x][y] = B[(xB + x) * N + yB + y];
            else sB[x][y] = 0;
        }
        __syncthreads();
        yA += BK;
        xB += BK;
#pragma unroll
        for (int j = 0; j < BK; j++) {
#pragma unroll
            for (int k = 0; k < TN; k++)
                ra[k] = sA[k + TN * thdx][j];
#pragma unroll
            for (int k = 0; k < TM; k++)
                rb[k] = sB[j][k + TM * thdy];
#pragma unroll
            for (int x = 0; x < TN; x++)
#pragma unroll
                for (int y = 0; y < TM; y++)
                    c[x][y] += ra[x] * rb[y];
        }
        __syncthreads();
    }
#pragma unroll
    for (int x = 0; x < TN; x++)
#pragma unroll
        for (int y = 0; y < TM; y++)
            if (xC + x < N && yC + y < N)
                C[(xC + x) * N + yC + y] = c[x][y];
}

__global__ void matMul(int N, double *__restrict__ C, const double *const __restrict__ A, const double *const __restrict__ B)
{
    if (N > 384)
        block_matMul<LG_BLOCK_N, LG_BLOCK_M, LG_BLOCK_K, LG_THREAD_N, LG_THREAD_M, LG_GRID_SIZE>(N, C, A, B);
    else
        block_matMul<SM_BLOCK_N, SM_BLOCK_M, SM_BLOCK_K, SM_THREAD_N, SM_THREAD_M, SM_GRID_SIZE>(N, C, A, B);
}
