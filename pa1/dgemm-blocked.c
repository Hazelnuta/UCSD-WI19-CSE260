/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <x86intrin.h>
#include <immintrin.h>
#include <string.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
// #define BLOCK_SIZE 37
// #define BLOCK_SIZE 719
#endif


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C){
	/* For each row i of A */
    for (int i = 0; i < M; ++i)
		/* For each column j of B */
        for (int j = 0; j < N; ++j) {
			/* Compute C(i,j) */
            double cij = C[i * lda + j];
            for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
                cij += A[i * lda + k] * B[j * lda + k];
#else
                cij += A[i * lda + k] * B[k * lda + j];
#endif
            C[i * lda + j] = cij;
        }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void orig_square_dgemm(int lda, double* A, double* B, double* C){
#ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
        for (int j = i + 1; j < lda; ++j) {
            double t = B[i * lda + j];
            B[i * lda + j] = B[j * lda + i];
            B[j * lda + i] = t;
        }
#endif
	/* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
		/* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
				/* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

				/* Perform individual block dgemm */
#ifdef TRANSPOSE
                do_block(lda, M, N, K, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
                do_block(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
            }
#if TRANSPOSE
    for (int i = 0; i < lda; ++i)
        for (int j = i + 1; j < lda; ++j) {
            double t = B[i * lda + j];
            B[i * lda + j] = B[j * lda + i];
            B[j * lda + i] = t;
        }
#endif
}



#ifdef __GOTO__
#define KB 1024
#define L1_CACHE_SIZE (32 * KB)
#define L2_CACHE_SIZE (256 * KB)
#define L3_CACHE_SIZE (30 * 1024 * KB)
#ifndef NC
    #define NC 512
#endif
#ifndef KC
    #define KC 256
#endif
#ifndef MC
    #define MC 32
#endif
#ifndef MR
    #define MR 4
#endif
#ifndef NR
    #define NR 8
#endif
double nA[1064 * 1064] __attribute__((aligned(64)));
double pA[KC * MC] __attribute__((aligned(64)));
double nB[1064 * 1064] __attribute__((aligned(64)));
double pB[KC * NC] __attribute__((aligned(64)));
double nC[1064 * 1064] __attribute__((aligned(64)));
inline void mul_block_kernel(int lda, int nc, int kc, int mc,
                             double *  A, double *  B, double *  C){
    #if !defined(__AVX256__) && !defined(__AVX512__)
    for (int i = 0; i < kc; i++) {
        for (int j = 0; j < NR; j++) {
            for (int k = 0; k < MR; k++) {
                C[k * lda + j] += A[i * MR + k] * B[i * NR + j];
            }
        }
    }
    #elif defined(__AVX512__)
    __m512d a[MR], b[NR / 8], c[MR][NR / 8];

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR / 8; j++) {
            c[i][j] = _mm512_load_pd(C + i * lda + j * 8);
        }
    }
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < MR; i++) {
            a[i] = _mm512_broadcastsd_pd(_mm_set1_pd(A[k * MR + i]));
        }
        for (int j = 0; j < NR / 8; j++) {
            b[j] = _mm512_load_pd(B + k * NR + j * 8);
        }
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < NR / 8; j++) {
				// c[i][j] = _mm512_add_pd(c[i][j], _mm512_mul_pd(a[i], b[j]));
                c[i][j] = _mm512_fmadd_pd(a[i], b[j], c[i][j]);
            }
        }
    }
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR / 8; j++) {
            _mm512_store_pd(C + i * lda + j * 8, c[i][j]);
        }
    }
    #elif defined(__AVX256__)
    __m256d a[MR], b[NR / 4], c[MR][NR / 4];

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR / 4; j++) {
            c[i][j] = _mm256_load_pd(C + i * lda + j * 4);
        }
    }
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < MR; i++) {
            a[i] = _mm256_broadcast_sd(A + k * MR + i);
        }
        for (int j = 0; j < NR / 4; j++) {
            b[j] = _mm256_load_pd(B + k * NR + j * 4);
        }
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < NR / 4; j++) {
				// c[i][j] = _mm256_add_pd(c[i][j], _mm256_mul_pd(a[i], b[j]));
                c[i][j] = _mm256_fmadd_pd(a[i], b[j], c[i][j]);
            }
        }
    }
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR / 4; j++) {
            _mm256_store_pd(C + i * lda + j * 4, c[i][j]);
        }
    }
    #endif
}
inline void mul_block_loop1(int lda, int nc, int kc, int mc, double *  A, double *  B, double *  C){
    for (int i = 0; i < mc; i += MR) {
        mul_block_kernel(lda, nc, kc, mc, A + i * kc, B, C + i * lda);
    }
}
inline void mul_block_loop2(int lda, int nc, int kc, int mc, double *  A, double *  B, double *  C){
    for (int i = 0; i < nc; i += NR) {
        mul_block_loop1(lda, nc, kc, mc, A, B + i * kc, C + i);
    }
}
inline void mul_block_packa(int lda, int mc, int kc, double *  A){
    for (int k = 0; k < mc; k += MR) {
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < kc; j++) {
                pA[k * kc + j * MR + i] = A[(k + i) * lda + j];
            }
        }
    }
}
inline void mul_block_loop3(int lda, int nc, int kc, double *  A, double *  B, double *  C){
    for (int i = 0; i < lda; i += MC) {
        int mc = min(MC, lda - i);
        mul_block_packa(lda, mc, kc, A + i * lda);
        mul_block_loop2(lda, nc, kc, mc, pA, B, C + i * lda);
    }
}
inline void mul_block_packb(int lda, int nc, int kc, double *  B){
    for (int k = 0; k < nc; k += NR) {
        for (int i = 0; i < kc; i++) {
            for (int j = 0; j < NR; j++) {
                pB[k * kc + i * NR + j] = B[i * lda + k + j];
            }
        }
    }
}
inline void mul_block_loop4(int lda, int nc, double *  A, double *  B, double *  C){
    for (int i = 0; i < lda; i += KC) {
        int kc = min(KC, lda - i);
        mul_block_packb(lda, nc, kc, B + i * lda);
        mul_block_loop3(lda, nc, kc, A + i, pB, C);
    }
}
inline void mul_block_loop5(int lda, double *  A, double *  B, double *  C){
    for (int i = 0; i < lda; i += NC) {
        int nc = min(NC, lda - i);
        mul_block_loop4(lda, nc, A, B + i, C + i);
    }
}
inline void mul_block(int lda, double *  A, double *  B, double *  C){
	//padding
    int padding = max(NR, MR);
    int nlda = ((lda + padding - 1) / padding) * padding;

    memset(nA, 0, sizeof(double) * nlda * nlda);
    memset(nB, 0, sizeof(double) * nlda * nlda);
    for (int i = 0; i < lda; i++) {
        memcpy(nA + i * nlda, A + i * lda, sizeof(double) * lda);
        memcpy(nB + i * nlda, B + i * lda, sizeof(double) * lda);
        memcpy(nC + i * nlda, C + i * lda, sizeof(double) * lda);
    }
    int olda = lda;
    lda = nlda;

    mul_block_loop5(lda, nA, nB, nC);

    for (int i = 0; i < olda; i++) {
        memcpy(C + i * olda, nC + i * lda, sizeof(double) * olda);
    }
}
#endif

#ifdef __LEVEL__
double nA[1064 * 1064] __attribute__ ((aligned(64)));
double nB[1064 * 1064] __attribute__ ((aligned(64)));
double nC[1064 * 1064] __attribute__ ((aligned(64)));
#ifndef block_size_sm
    #define block_size_sm 32
#endif
#ifndef block_size_l1
    #define block_size_l1 64
#endif
#ifndef block_size_l2
    #define block_size_l2 256
#endif

double pA[block_size_sm * block_size_sm] __attribute__ ((aligned(32)));
double pB[block_size_sm * block_size_sm] __attribute__ ((aligned(32)));
inline void mul_block_sm(int lda, double *  A, double *  B, double *  C){
    for (int i = 0; i < block_size_sm; i++) {
        for (int j = 0; j < block_size_sm; j++) {
            __m256d a[block_size_sm / 4];
            for (int k = 0; k < block_size_sm; k += 4) {
                a[k / 4] = _mm256_load_pd(&A[i * block_size_sm + k]);
                a[k / 4] = _mm256_mul_pd(a[k / 4], _mm256_load_pd(&B[j * block_size_sm + k]));
            }
            for (int k = 1; k < block_size_sm / 4; k++) {
                a[0] = _mm256_add_pd(a[0], a[k]);
            }
            a[0] = _mm256_hadd_pd(a[0], a[0]);
            C[i * lda + j] += ((double*)(&a[0]))[0] + ((double*)(&a[0]))[2];
        }
    }
}

inline void mul_block_any(int lda, int N, int M, int K, double *  A, double *  B, double *  C){
    for (int i = 0; i < N; i++) {
        int ilda = i * lda;
        for (int j = 0; j < M; j++) {
            register double c = C[ilda + j];
            for (int k = 0; k < K; k++) {
                c += A[i * N + k] * B[k * M + j];
            }
            C[ilda + j] = c;
        }
    }
}
inline void mul_block_l1(int lda, int N, int M, int K, double *  A, double *  B, double *  C){
    for (int i = 0; i < N; i += block_size_sm) {
        int ilda = i * lda;
        int NN = min(block_size_sm, N - i);
        for (int k = 0; k < K; k += block_size_sm) {
            int KK = min(block_size_sm, K - k);
            for (int r = 0; r < NN; r++) {
                for (int c = 0; c < KK; c++) {
                    pA[r * NN + c] = A[(i + r) * lda + k + c];
                }
            }
            for (int j = 0; j < M; j += block_size_sm) {
                int MM = min(block_size_sm, M - j);
                for (int r = 0; r < MM; r++) {
                    for (int c = 0; c < KK; c++) {
                        pB[c * MM + r] = B[(k + r) * lda + j + c];
                    }
                }
                #ifdef __AVX256__
                if (NN == block_size_sm && MM == block_size_sm && KK == block_size_sm) {
                    mul_block_sm(lda, pA, pB, C + ilda + j);
                }else{
                    mul_block_any(lda, NN, MM, KK, pA, pB, C + ilda + j);
                }
                #else
                mul_block_any(lda, NN, MM, KK, pA, pB, C + ilda + j);
                #endif
            }
        }
    }
}
inline void mul_block_l2(int lda, int N, int M, int K, double *  A, double *  B, double *  C){
    for (int i = 0; i < N; i += block_size_l1) {
        int ilda = i * lda;
        for (int k = 0; k < K; k += block_size_l1) {
            int klda = k * lda;
            for (int j = 0; j < M; j += block_size_l1) {
                int NN = min(N - i, block_size_l1);
                int MM = min(M - j, block_size_l1);
                int KK = min(K - k, block_size_l1);
                mul_block_l1(lda, NN, MM, KK, A + ilda + k, B + klda + j, C + ilda + j);
            }
        }
    }
}
inline void mul_block(int lda, double * A, double *  B, double *  C){
    int nlda = ((lda + 3) / 4) * 4;

    memset(nA, 0, sizeof(double) * nlda * nlda);
    memset(nB, 0, sizeof(double) * nlda * nlda);

    for (int i = 0; i < lda; i++) {
        memcpy(nA + i * nlda, A + i * lda, sizeof(double) * lda);
        memcpy(nB + i * nlda, B + i * lda, sizeof(double) * lda);
    }
    int olda = lda;

    lda = nlda;

    for (int i = 0; i < lda; ++i)
        for (int j = i + 1; j < lda; ++j) {
            double t = nB[i * lda + j];
            nB[i * lda + j] = nB[j * lda + i];
            nB[j * lda + i] = t;
        }
    for (int i = 0; i < lda; i += block_size_l2) {
        int ilda = i * lda;
        for (int k = 0; k < lda; k += block_size_l2) {
            int klda = k * lda;
            for (int j = 0; j < lda; j += block_size_l2) {
                int N = min(lda - i, block_size_l2);
                int M = min(lda - j, block_size_l2);
                int K = min(lda - k, block_size_l2);
                mul_block_l2(lda, N, M, K, A + ilda + k, B + klda + j, C + ilda + j);
            }
        }
    }
    for (int i = 0; i < olda; i++) {
        memcpy(C + i * olda, nC + i * lda, sizeof(double) * olda);
    }
}
#endif


void square_dgemm(int lda, double* A, double* B, double* C){
    #if defined(__GOTO__) || defined(__LEVEL__)
    mul_block(lda, A, B, C);
    #else
    orig_square_dgemm(lda, A, B, C);
    #endif
}
