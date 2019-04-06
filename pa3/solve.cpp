/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;
#ifndef FUSED
#define FUSED
#endif
void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], int r, double *E, int m, int n);

extern control_block cb;

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq)
{
    double l2norm = sumSq / (double)((cb.m) * (cb.n));

    l2norm = sqrt(l2norm);
    return l2norm;
}
#ifdef _MPI_
// #define COL_LENGTH (cols + 2 + (cols & 1))
#define COL_LENGTH (cols + 2)
#define ID(x, y) ((x) * COL_LENGTH + (y))
enum {
    TOP=0,
    BOTTOM,
    LEFT,
    RIGHT
};
int procs, rankid, rankx, ranky;
int rows, cols;
double in_L[8010], in_R[8010];
double out_L[8010], out_R[8010];
void communicate(double *E_prev)
{
	// printf("comm %d %d %d\n", rankid, rankx, ranky);
    MPI_Request recv_requests[4], send_requests[4];
    MPI_Status statuses[4];
    int recv_cnt = 0;

	// top
    if (rankx == 0) {
        for (int i = 1; i <= cols; i++)
            E_prev[ID(0, i)] = E_prev[ID(2, i)];
    } else {
        MPI_Isend(&E_prev[ID(1, 1)], cols, MPI_DOUBLE, rankid - cb.py, BOTTOM, MPI_COMM_WORLD, send_requests + 0);
        MPI_Irecv(&E_prev[ID(0, 1)], cols, MPI_DOUBLE, rankid - cb.py, TOP, MPI_COMM_WORLD, recv_requests + (recv_cnt++));
    }
	// bottom
    if (rankx == cb.px - 1) {
        for (int i = 1; i <= cols; i++)
            E_prev[ID(rows + 1, i)] = E_prev[ID(rows - 1, i)];
    } else {
        MPI_Isend(&E_prev[ID(rows, 1)], cols, MPI_DOUBLE, rankid + cb.py, TOP, MPI_COMM_WORLD, send_requests + 1);
        MPI_Irecv(&E_prev[ID(rows + 1, 1)], cols, MPI_DOUBLE, rankid + cb.py, BOTTOM, MPI_COMM_WORLD, recv_requests + (recv_cnt++));
    }
	// left
    if (ranky == 0) {
        for (int i = 1; i <= rows; i++)
            E_prev[ID(i, 0)] = E_prev[ID(i, 2)];
		// printf("left %f\n", E_prev[ID(i, 0)]);
    } else {
        for (int i = 1; i <= rows; i++)
            out_L[i] = E_prev[ID(i, 1)];
		// printf("sendL %d %d %f\n", rankid, i, out_L[i]);

        MPI_Isend(&out_L[1], rows, MPI_DOUBLE, rankid - 1, RIGHT, MPI_COMM_WORLD, send_requests + 2);
        MPI_Irecv(&in_L[1], rows, MPI_DOUBLE, rankid - 1, LEFT, MPI_COMM_WORLD, recv_requests + (recv_cnt++));
    }
	// right
    if (ranky == cb.py - 1) {
        for (int i = 1; i <= rows; i++)
            E_prev[ID(i, cols + 1)] = E_prev[ID(i, cols - 1)];
    } else {
        for (int i = 1; i <= rows; i++)
            out_R[i] = E_prev[ID(i, cols)];
		// printf("sendR %d %d %f\n", rankid, i, out_R[i]);
        MPI_Isend(&out_R[1], rows, MPI_DOUBLE, rankid + 1, LEFT, MPI_COMM_WORLD, send_requests + 3);
        MPI_Irecv(&in_R[1], rows, MPI_DOUBLE, rankid + 1, RIGHT, MPI_COMM_WORLD, recv_requests + (recv_cnt++));
    }
    MPI_Waitall(recv_cnt, recv_requests, statuses);
    if (ranky != cb.py - 1)
        for (int i = 1; i <= rows; i++)
            E_prev[ID(i, cols + 1)] = in_R[i];
	// printf("recvR %d %d %f\n", rankid, i, in_R[i]);

    if (ranky != 0)
        for (int i = 1; i <= rows; i++)
            E_prev[ID(i, 0)] = in_L[i];
	// printf("recvL %d %d %f\n", rankid, i, in_L[i]);
}

void solve_MPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{
    int m = cb.m, n = cb.n;

    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
    rankx = rankid / cb.py;
    ranky = rankid % cb.py;
    rows = m / cb.px;
    rows += (rankx < m - rows * cb.px);
    cols = n / cb.py;
    cols += (ranky < n - cols * cb.py);
	// Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int innerBlockRowStartIndex = ID(1, 1);
    int innerBlockRowEndIndex = ID(rows, cols);
	// printf("=== %d %d %d %d %d %d %d\n", rankid, rankx, ranky, rows, cols, innerBlockRowStartIndex, innerBlockRowEndIndex);
#ifdef SSE_VEC
    __m128d alpha_2 = _mm_set1_pd(alpha);
    __m128d four_2 = _mm_set1_pd(4);
    __m128d one_2 = _mm_set1_pd(1);
    __m128d none_2 = _mm_set1_pd(-1);
    __m128d M1_2 = _mm_set1_pd(M1);
    __m128d M2_2 = _mm_set1_pd(M2);
    __m128d a_2 = _mm_set1_pd(a);
    __m128d b_2 = _mm_set1_pd(b);
    __m128d dt_2 = _mm_set1_pd(dt);
    __m128d kk_2 = _mm_set1_pd(kk);
    __m128d epsilon_2 = _mm_set1_pd(epsilon);
#endif
    for (niter = 0; niter < cb.niters; niter++) {
		// printf("===== %d %d\n", rankid, niter);
#ifndef PRODUCTION
        if (cb.debug && (niter == 0)) {
            stats(E_prev, rows, cols, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, rows, cols, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, rows, cols);
        }
#endif
        if (!cb.noComm)
            communicate(E_prev);

        int i, j;
#ifdef SSE_VEC
#define STEP 2
#else
#define STEP 1
#endif
#ifdef FUSED
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += COL_LENGTH) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < cols; i += STEP) {
#ifdef SSE_VEC
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d ET_2 = _mm_loadu_pd(&E_prev_tmp[i - COL_LENGTH]);
                __m128d EB_2 = _mm_loadu_pd(&E_prev_tmp[i + COL_LENGTH]);
                __m128d EL_2 = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                __m128d ER_2 = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                __m128d E_2 = _mm_add_pd(EC_2,
                                         _mm_mul_pd(alpha_2,
                                                    _mm_sub_pd(_mm_add_pd(_mm_add_pd(ET_2, EB_2),
                                                                          _mm_add_pd(EL_2, ER_2)),
                                                               _mm_mul_pd(four_2, EC_2))));
				// __m128d E_2 = _mm_load_pd(&E_tmp[i]);
				// __m128d EC_2 = _mm_load_pd(&E_prev_tmp[i]);
                __m128d R_2 = _mm_loadu_pd(&R_tmp[i]);
                E_2 = _mm_sub_pd(E_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_add_pd(_mm_mul_pd(EC_2, R_2),
                                                       _mm_mul_pd(_mm_mul_pd(_mm_sub_pd(EC_2, a_2),
                                                                             _mm_sub_pd(EC_2, one_2)),
                                                                  _mm_mul_pd(kk_2, EC_2)))));
                R_2 = _mm_add_pd(R_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_mul_pd(_mm_add_pd(epsilon_2,
                                                                  _mm_div_pd(_mm_mul_pd(M1_2, R_2),
                                                                             _mm_add_pd(EC_2, M2_2))),
                                                       _mm_sub_pd(_mm_mul_pd(none_2, R_2),
                                                                  _mm_mul_pd(kk_2,
                                                                             _mm_mul_pd(EC_2,
                                                                                        _mm_sub_pd(EC_2, _mm_add_pd(b_2, one_2))))))));
                _mm_storeu_pd(&E_tmp[i], E_2);
                _mm_storeu_pd(&R_tmp[i], R_2);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + COL_LENGTH] + E_prev_tmp[i - COL_LENGTH]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#else
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += COL_LENGTH) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < cols; i += STEP) {
#ifdef SSE_VEC
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d ET_2 = _mm_loadu_pd(&E_prev_tmp[i - COL_LENGTH]);
                __m128d EB_2 = _mm_loadu_pd(&E_prev_tmp[i + COL_LENGTH]);
                __m128d EL_2 = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                __m128d ER_2 = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                __m128d E_2 = _mm_add_pd(EC_2,
                                         _mm_mul_pd(alpha_2,
                                                    _mm_sub_pd(_mm_add_pd(_mm_add_pd(ET_2, EB_2),
                                                                          _mm_add_pd(EL_2, ER_2)),
                                                               _mm_mul_pd(four_2, EC_2))));
                _mm_storeu_pd(&E_tmp[i], E_2);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + COL_LENGTH] + E_prev_tmp[i - COL_LENGTH]);
#endif
            }
            for (i = 0; i < cols; i += STEP) {
#ifdef SSE_VEC
                __m128d E_2 = _mm_loadu_pd(&E_tmp[i]);
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d R_2 = _mm_loadu_pd(&R_tmp[i]);
                E_2 = _mm_sub_pd(E_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_add_pd(_mm_mul_pd(EC_2, R_2),
                                                       _mm_mul_pd(_mm_mul_pd(_mm_sub_pd(EC_2, a_2),
                                                                             _mm_sub_pd(EC_2, one_2)),
                                                                  _mm_mul_pd(kk_2, EC_2)))));
                R_2 = _mm_add_pd(R_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_mul_pd(_mm_add_pd(epsilon_2,
                                                                  _mm_div_pd(_mm_mul_pd(M1_2, R_2),
                                                                             _mm_add_pd(EC_2, M2_2))),
                                                       _mm_sub_pd(_mm_mul_pd(none_2, R_2),
                                                                  _mm_mul_pd(kk_2,
                                                                             _mm_mul_pd(EC_2,
                                                                                        _mm_sub_pd(EC_2, _mm_add_pd(b_2, one_2))))))));
                _mm_storeu_pd(&E_tmp[i], E_2);
                _mm_storeu_pd(&R_tmp[i], R_2);
#else
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#endif
#ifndef PRODUCTION
        if (cb.stats_freq) {
            if (!(niter % cb.stats_freq)) {
                stats(E, rows, cols, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, rows, cols, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq)
            if (!(niter % cb.plot_freq))
                plotter->updatePlot(E, niter, rows, cols);
#endif

		// Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;
    } //end of 'niter' loop at the beginning
	// printMat2("Rank 0 Matrix E_prev", rankid, E_prev, rows, cols); // return the L2 and infinity norms via in-out parameters
    stats(E_prev, rows, cols, &Linf, &sumSq);
	// printf("stat %d\n", rankid);
    MPI_Barrier(MPI_COMM_WORLD);
    double _Linf, _sumSq;
    MPI_Reduce(&Linf, &_Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumSq, &_sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    Linf = _Linf;
    sumSq = _sumSq;
    L2 = L2Norm(sumSq);

	// Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}
#endif
void solve_single(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{
	// Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n = cb.n;
    int innerBlockRowStartIndex = (n + 2) + 1;
    int innerBlockRowEndIndex = (((m + 2) * (n + 2) - 1) - (n)) - (n + 2);

#ifdef SSE_VEC
    __m128d alpha_2 = _mm_set1_pd(alpha);
    __m128d four_2 = _mm_set1_pd(4);
    __m128d one_2 = _mm_set1_pd(1);
    __m128d none_2 = _mm_set1_pd(-1);
    __m128d M1_2 = _mm_set1_pd(M1);
    __m128d M2_2 = _mm_set1_pd(M2);
    __m128d a_2 = _mm_set1_pd(a);
    __m128d b_2 = _mm_set1_pd(b);
    __m128d dt_2 = _mm_set1_pd(dt);
    __m128d kk_2 = _mm_set1_pd(kk);
    __m128d epsilon_2 = _mm_set1_pd(epsilon);
#endif

	// We continue to sweep over the mesh until the simulation has reached
	// the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++) {
        if (cb.debug && (niter == 0)) {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m + 1, n + 1);
        }

		/*
		 * Copy data from boundary of the computational box to the
		 * padding region, set up for differencing computational box's boundary
		 *
		 * These are physical boundary conditions, and are not to be confused
		 * with ghost cells that we would use in an MPI implementation
		 *
		 * The reason why we copy boundary conditions is to avoid
		 * computing single sided differences at the boundaries
		 * which increase the running time of solve()
		 *
		 */

		// 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i, j;

		// Fills in the TOP Ghost Cells
        for (i = 0; i < (n + 2); i++)
            E_prev[i] = E_prev[i + (n + 2) * 2];

		// Fills in the RIGHT Ghost Cells
        for (i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2))
            E_prev[i] = E_prev[i - 2];

		// Fills in the LEFT Ghost Cells
        for (i = 0; i < (m + 2) * (n + 2); i += (n + 2))
            E_prev[i] = E_prev[i + 2];

		// Fills in the BOTTOM Ghost Cells
        for (i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++)
            E_prev[i] = E_prev[i - (n + 2) * 2];

//////////////////////////////////////////////////////////////////////////////
#define COL_LENGTH (n + 2)
#ifdef SSE_VEC
#define STEP 2
#else
#define STEP 1
#endif
#ifdef FUSED
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += COL_LENGTH) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < n; i += STEP) {
#ifdef SSE_VEC
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d ET_2 = _mm_loadu_pd(&E_prev_tmp[i - COL_LENGTH]);
                __m128d EB_2 = _mm_loadu_pd(&E_prev_tmp[i + COL_LENGTH]);
                __m128d EL_2 = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                __m128d ER_2 = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                __m128d E_2 = _mm_add_pd(EC_2,
                                         _mm_mul_pd(alpha_2,
                                                    _mm_sub_pd(_mm_add_pd(_mm_add_pd(ET_2, EB_2),
                                                                          _mm_add_pd(EL_2, ER_2)),
                                                               _mm_mul_pd(four_2, EC_2))));
				// __m128d E_2 = _mm_load_pd(&E_tmp[i]);
				// __m128d EC_2 = _mm_load_pd(&E_prev_tmp[i]);
                __m128d R_2 = _mm_loadu_pd(&R_tmp[i]);
                E_2 = _mm_sub_pd(E_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_add_pd(_mm_mul_pd(EC_2, R_2),
                                                       _mm_mul_pd(_mm_mul_pd(_mm_sub_pd(EC_2, a_2),
                                                                             _mm_sub_pd(EC_2, one_2)),
                                                                  _mm_mul_pd(kk_2, EC_2)))));
                R_2 = _mm_add_pd(R_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_mul_pd(_mm_add_pd(epsilon_2,
                                                                  _mm_div_pd(_mm_mul_pd(M1_2, R_2),
                                                                             _mm_add_pd(EC_2, M2_2))),
                                                       _mm_sub_pd(_mm_mul_pd(none_2, R_2),
                                                                  _mm_mul_pd(kk_2,
                                                                             _mm_mul_pd(EC_2,
                                                                                        _mm_sub_pd(EC_2, _mm_add_pd(b_2, one_2))))))));
                _mm_storeu_pd(&E_tmp[i], E_2);
                _mm_storeu_pd(&R_tmp[i], R_2);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + COL_LENGTH] + E_prev_tmp[i - COL_LENGTH]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#else
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += COL_LENGTH) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i += STEP) {
#ifdef SSE_VEC
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d ET_2 = _mm_loadu_pd(&E_prev_tmp[i - COL_LENGTH]);
                __m128d EB_2 = _mm_loadu_pd(&E_prev_tmp[i + COL_LENGTH]);
                __m128d EL_2 = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                __m128d ER_2 = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                __m128d E_2 = _mm_add_pd(EC_2,
                                         _mm_mul_pd(alpha_2,
                                                    _mm_sub_pd(_mm_add_pd(_mm_add_pd(ET_2, EB_2),
                                                                          _mm_add_pd(EL_2, ER_2)),
                                                               _mm_mul_pd(four_2, EC_2))));
                _mm_storeu_pd(&E_tmp[i], E_2);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + COL_LENGTH] + E_prev_tmp[i - COL_LENGTH]);
#endif
            }
            for (i = 0; i < n; i += STEP) {
#ifdef SSE_VEC
                __m128d E_2 = _mm_loadu_pd(&E_tmp[i]);
                __m128d EC_2 = _mm_loadu_pd(&E_prev_tmp[i]);
                __m128d R_2 = _mm_loadu_pd(&R_tmp[i]);
                E_2 = _mm_sub_pd(E_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_add_pd(_mm_mul_pd(EC_2, R_2),
                                                       _mm_mul_pd(_mm_mul_pd(_mm_sub_pd(EC_2, a_2),
                                                                             _mm_sub_pd(EC_2, one_2)),
                                                                  _mm_mul_pd(kk_2, EC_2)))));
                R_2 = _mm_add_pd(R_2,
                                 _mm_mul_pd(dt_2,
                                            _mm_mul_pd(_mm_add_pd(epsilon_2,
                                                                  _mm_div_pd(_mm_mul_pd(M1_2, R_2),
                                                                             _mm_add_pd(EC_2, M2_2))),
                                                       _mm_sub_pd(_mm_mul_pd(none_2, R_2),
                                                                  _mm_mul_pd(kk_2,
                                                                             _mm_mul_pd(EC_2,
                                                                                        _mm_sub_pd(EC_2, _mm_add_pd(b_2, one_2))))))));
                _mm_storeu_pd(&E_tmp[i], E_2);
                _mm_storeu_pd(&R_tmp[i], R_2);
#else
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#endif
		/////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq) {
            if (!(niter % cb.stats_freq)) {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq)
            if (!(niter % cb.plot_freq))
                plotter->updatePlot(E, niter, m, n);

		// Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;
    } //end of 'niter' loop at the beginning

    printMat2("Rank 0 Matrix E_prev", 0, E_prev, m, n); // return the L2 and infinity norms via in-out parameters

    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

	// Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}

void printMat2(const char mesg[], int r, double *E, int m, int n)
{
    int i;

#if 0
    if (m > 8)
        return;
#else
    if (m > 34)
        return;
#endif
    printf("%s %d\n", mesg, r);
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


void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{
#ifdef _MPI_
    solve_MPI(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#else
    solve_single(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#endif
}
