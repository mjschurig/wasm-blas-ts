/**
 * DSYR2K - Double precision symmetric rank-2k update
 * 
 * Computes: C := alpha*A*B^T + alpha*B*A^T + beta*C  or
 *           C := alpha*A^T*B + alpha*B^T*A + beta*C
 * where C is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 3 DSYR2K routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part  
 * @param trans  'N': C := alpha*A*B^T + alpha*B*A^T + beta*C, 'T'/'C': C := alpha*A^T*B + alpha*B^T*A + beta*C
 * @param n      Order of matrix C
 * @param k      Number of columns of A and B (if trans='N') or rows of A and B (if trans='T')
 * @param alpha  Scalar multiplier for A*B^T + B*A^T or A^T*B + B^T*A
 * @param a      Matrix A
 * @param lda    Leading dimension of A
 * @param b      Matrix B  
 * @param ldb    Leading dimension of B
 * @param beta   Scalar multiplier for C
 * @param c      Input/output symmetric matrix C (n x n)
 * @param ldc    Leading dimension of C
 */

extern "C" {

void dsyr2k(char uplo, char trans, int n, int k, double alpha,
            const double* a, int lda, const double* b, int ldb,
            double beta, double* c, int ldc) {
    
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
        // Form C := alpha*A*B^T + alpha*B*A^T + beta*C
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
                    if (a[j + l * lda] != zero || b[j + l * ldb] != zero) {
                        double temp1 = alpha * b[j + l * ldb];
                        double temp2 = alpha * a[j + l * lda];
                        for (int i = 0; i <= j; i++) {
                            c[i + j * ldc] += a[i + l * lda] * temp1 + b[i + l * ldb] * temp2;
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
                    if (a[j + l * lda] != zero || b[j + l * ldb] != zero) {
                        double temp1 = alpha * b[j + l * ldb];
                        double temp2 = alpha * a[j + l * lda];
                        for (int i = j; i < n; i++) {
                            c[i + j * ldc] += a[i + l * lda] * temp1 + b[i + l * ldb] * temp2;
                        }
                    }
                }
            }
        }
    } else {
        // Form C := alpha*A^T*B + alpha*B^T*A + beta*C
        if (upper) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                    double temp1 = zero;
                    double temp2 = zero;
                    for (int l = 0; l < k; l++) {
                        temp1 += a[l + i * lda] * b[l + j * ldb];
                        temp2 += b[l + i * ldb] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp1 + alpha * temp2;
                    } else {
                        c[i + j * ldc] = alpha * temp1 + alpha * temp2 + beta * c[i + j * ldc];
                    }
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    double temp1 = zero;
                    double temp2 = zero;
                    for (int l = 0; l < k; l++) {
                        temp1 += a[l + i * lda] * b[l + j * ldb];
                        temp2 += b[l + i * ldb] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp1 + alpha * temp2;
                    } else {
                        c[i + j * ldc] = alpha * temp1 + alpha * temp2 + beta * c[i + j * ldc];
                    }
                }
            }
        }
    }
}

} // extern "C"
