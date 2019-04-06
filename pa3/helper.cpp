/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;
extern control_block cb;

void printMat(const char mesg[], double *E, int m, int n);



//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
// #define COL_LENGTH (cols + 2 + (cols & 1))
#define COL_LENGTH (cols + 2)
#define ID(x, y) ((x) * COL_LENGTH + (y))
void init(double *E, double *E_prev, double *R, int m, int n)
{
    int procs, rankid;
    int rankx, ranky;
    int rows, cols;

#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
#else
    rankid = 0;
#endif
    rankx = rankid / cb.py;
    ranky = rankid % cb.py;
    rows = m / cb.px;
    rows += (rankx < m - rows * cb.px);
    cols = n / cb.py;
    cols += (ranky < n - cols * cb.py);
    int i;
    for (int i = 0; i < rows + 2; i++)
        for (int j = 0; j < cols + 2; j++)
            E_prev[ID(i, j)] = R[ID(i, j)] = 0;
	// printf("fill0 %d\n", rankid);
	// for (i = 0; i < (rows + 2) * (cols + 2); i++)
	//     E_prev[i] = R[i] = 0;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            int globalx, globaly;
            globalx = (m / cb.px) * rankx + min(m - (m / cb.px) * cb.px, rankx) + i;
            globaly = (n / cb.py) * ranky + min(n - (n / cb.py) * cb.py, ranky) + j;
            if (globaly != n + 1 && globaly >= (n + 1) / 2 + 1)
                E_prev[ID(i, j)] = 1;
        }
    }
	// printf("fill ep %d\n", rankid);

	// for (i = (n + 2); i < (m + 1) * (n + 2); i++) {
	//     int colIndex = i % (n + 2); // gives the base index (first row's) of the current index
	//
	//  // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	//     if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
	//         continue;
	//
	//     E_prev[i] = 1.0;
	// }

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            int globalx, globaly;
            globalx = (m / cb.px) * rankx + min(m - (m / cb.px) * cb.px, rankx) + i;
            globaly = (n / cb.py) * ranky + min(n - (n / cb.py) * cb.py, ranky) + j;
            if (globaly != n + 1 && globaly != 0 && globalx >= (m + 1) / 2 + 1)
                R[ID(i, j)] = 1;
        }
    }
	// printf("fillr %d\n", rankid);
	// for (i = 0; i < (m + 2) * (n + 2); i++) {
	//     int rowIndex = i / (n + 2); // gives the current row number in 2D array representation
	//     int colIndex = i % (n + 2); // gives the base index (first row's) of the current index
	//
	//  // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	//     if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
	//         continue;
	//
	//     R[i] = 1.0;
	// }
	// We only print the meshes if they are small enough
#if 0
    printMat("E_prev", E_prev, rows, cols);
    printMat("R", R, rows, cols);
#endif
}

double *alloc1D(int m, int n)
{
    int procs, rankid;
    int rankx, ranky;
    int rows, cols;

#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
#else
    rankid = 0;
#endif
    rankx = rankid / cb.py;
    ranky = rankid % cb.py;
    rows = m / cb.px;
    rows += (rankx < m - rows * cb.px);
    cols = n / cb.py;
    cols += (ranky < n - cols * cb.py);
    int nx = cols + 40, ny = rows + 40;
	// int nx = n, ny = m;
	// printf("%d %d %d\n", rankid, nx, ny);
    double *E;

	// Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return E;
}

void printMat(const char mesg[], double *E, int m, int n)
{
    int i;

#if 0
    if (m > 8)
        return;
#else
    if (m > 34)
        return;
#endif
    int rankid;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
#else
    rankid = 0;
#endif
    printf("%s %d\n", mesg, rankid);
    for (i = 0; i < (m + 2) * (n + 2); i++) {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex > 0) && (colIndex < n + 1))
            if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
