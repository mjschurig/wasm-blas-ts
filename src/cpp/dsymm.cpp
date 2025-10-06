/**
 * DSYMM - Double precision symmetric matrix-matrix multiplication
 * 
 * Computes: C := alpha * A * B + beta * C  or  C := alpha * B * A + beta * C
 * where A is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 3 DSYMM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param side   'L': C := alpha*A*B + beta*C, 'R': C := alpha*B*A + beta*C
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part
 * @param m      Number of rows of matrix C
 * @param n      Number of columns of matrix C
 * @param alpha  Scalar multiplier for A*B or B*A
 * @param a      Symmetric matrix A
 * @param lda    Leading dimension of A
 * @param b      Matrix B
 * @param ldb    Leading dimension of B
 * @param beta   Scalar multiplier for C
 * @param c      Input/output matrix C
 * @param ldc    Leading dimension of C
 */

extern "C" {

void dsymm(char side, char uplo, int m, int n, double alpha,
           const double* a, int lda, const double* b, int ldb,
           double beta, double* c, int ldc) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    bool left = (side == 'L' || side == 'l');
    bool upper = (uplo == 'U' || uplo == 'u');
    
    // Quick return if possible
    if (m == 0 || n == 0 || ((alpha == zero || (left ? m : n) == 0) && beta == one)) {
        return;
    }
    
    // Handle beta
    if (alpha == zero) {
        if (beta == zero) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = zero;
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = beta * c[i + j * ldc];
                }
            }
        }
        return;
    }
    
    // Start the operations
    if (left) {
        // Form C := alpha*A*B + beta*C
        if (upper) {
            // Form C when A is upper triangular
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    double temp1 = alpha * b[i + j * ldb];
                    double temp2 = zero;
                    for (int k = 0; k < i; k++) {
                        c[k + j * ldc] += temp1 * a[k + i * lda];
                        temp2 += a[k + i * lda] * b[k + j * ldb];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = temp1 * a[i + i * lda] + alpha * temp2;
                    } else {
                        c[i + j * ldc] = beta * c[i + j * ldc] + temp1 * a[i + i * lda] + alpha * temp2;
                    }
                }
            }
        } else {
            // Form C when A is lower triangular
            for (int j = 0; j < n; j++) {
                for (int i = m - 1; i >= 0; i--) {
                    double temp1 = alpha * b[i + j * ldb];
                    double temp2 = zero;
                    for (int k = i + 1; k < m; k++) {
                        c[k + j * ldc] += temp1 * a[k + i * lda];
                        temp2 += a[k + i * lda] * b[k + j * ldb];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = temp1 * a[i + i * lda] + alpha * temp2;
                    } else {
                        c[i + j * ldc] = beta * c[i + j * ldc] + temp1 * a[i + i * lda] + alpha * temp2;
                    }
                }
            }
        }
    } else {
        // Form C := alpha*B*A + beta*C
        for (int j = 0; j < n; j++) {
            double temp1 = alpha * a[j + j * lda];
            if (beta == zero) {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = temp1 * b[i + j * ldb];
                }
            } else {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = beta * c[i + j * ldc] + temp1 * b[i + j * ldb];
                }
            }
            for (int k = 0; k < j; k++) {
                if (upper) {
                    temp1 = alpha * a[k + j * lda];
                } else {
                    temp1 = alpha * a[j + k * lda];
                }
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] += temp1 * b[i + k * ldb];
                }
            }
            for (int k = j + 1; k < n; k++) {
                if (upper) {
                    temp1 = alpha * a[j + k * lda];
                } else {
                    temp1 = alpha * a[k + j * lda];
                }
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] += temp1 * b[i + k * ldb];
                }
            }
        }
    }
}

} // extern "C"
