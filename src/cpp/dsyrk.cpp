/**
 * DSYRK - Double precision symmetric rank-k update
 * 
 * Computes: C := alpha * A * A^T + beta * C  or  C := alpha * A^T * A + beta * C
 * where C is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 3 DSYRK routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part  
 * @param trans  'N': C := alpha*A*A^T + beta*C, 'T'/'C': C := alpha*A^T*A + beta*C
 * @param n      Order of matrix C
 * @param k      Number of columns of A (if trans='N') or rows of A (if trans='T')
 * @param alpha  Scalar multiplier for A*A^T or A^T*A
 * @param a      Matrix A
 * @param lda    Leading dimension of A
 * @param beta   Scalar multiplier for C
 * @param c      Input/output symmetric matrix C (n x n)
 * @param ldc    Leading dimension of C
 */

extern "C" {

void dsyrk(char uplo, char trans, int n, int k, double alpha,
           const double* a, int lda, double beta, double* c, int ldc) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    bool notrans = (trans == 'N' || trans == 'n');
    
    // Quick return if possible
    if (n == 0 || ((alpha == zero || k == 0) && beta == one)) {
        return;
    }
    
    // Handle beta
    if (alpha == zero) {
        if (upper) {
            if (beta == zero) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i <= j; i++) {
                        c[i + j * ldc] = zero;
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i <= j; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
            }
        } else {
            if (beta == zero) {
                for (int j = 0; j < n; j++) {
                    for (int i = j; i < n; i++) {
                        c[i + j * ldc] = zero;
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    for (int i = j; i < n; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
            }
        }
        return;
    }
    
    // Start the operations
    if (notrans) {
        // Form C := alpha*A*A^T + beta*C
        if (upper) {
            for (int j = 0; j < n; j++) {
                if (beta == zero) {
                    for (int i = 0; i <= j; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (int i = 0; i <= j; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                for (int l = 0; l < k; l++) {
                    if (a[j + l * lda] != zero) {
                        double temp = alpha * a[j + l * lda];
                        for (int i = 0; i <= j; i++) {
                            c[i + j * ldc] += temp * a[i + l * lda];
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                if (beta == zero) {
                    for (int i = j; i < n; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (int i = j; i < n; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                for (int l = 0; l < k; l++) {
                    if (a[j + l * lda] != zero) {
                        double temp = alpha * a[j + l * lda];
                        for (int i = j; i < n; i++) {
                            c[i + j * ldc] += temp * a[i + l * lda];
                        }
                    }
                }
            }
        }
    } else {
        // Form C := alpha*A^T*A + beta*C
        if (upper) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                    double temp = zero;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp;
                    } else {
                        c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
                    }
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    double temp = zero;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp;
                    } else {
                        c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
                    }
                }
            }
        }
    }
}

} // extern "C"
