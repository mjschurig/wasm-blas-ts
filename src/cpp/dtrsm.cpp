/**
 * DTRSM - Double precision triangular solve with multiple right-hand sides
 * 
 * Solves: op(A)*X = alpha*B  or  X*op(A) = alpha*B
 * where op(A) = A or A^T, A is triangular, and X overwrites B
 * 
 * This is a C++ implementation of the BLAS Level 3 DTRSM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param side   'L': op(A)*X = alpha*B, 'R': X*op(A) = alpha*B
 * @param uplo   'U': upper triangular, 'L': lower triangular
 * @param transa 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param diag   'U': unit triangular, 'N': non-unit triangular
 * @param m      Number of rows of matrix B
 * @param n      Number of columns of matrix B
 * @param alpha  Scalar multiplier
 * @param a      Triangular matrix A
 * @param lda    Leading dimension of A
 * @param b      Input matrix B, output matrix X
 * @param ldb    Leading dimension of B
 */

extern "C" {

void dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha,
           const double* a, int lda, double* b, int ldb) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    bool left = (side == 'L' || side == 'l');
    bool upper = (uplo == 'U' || uplo == 'u');
    bool notrans = (transa == 'N' || transa == 'n');
    bool nounit = (diag == 'N' || diag == 'n');
    
    // Quick return if possible
    if (m == 0 || n == 0) return;
    
    // Handle alpha
    if (alpha == zero) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                b[i + j * ldb] = zero;
            }
        }
        return;
    }
    
    // Start the operations
    if (left) {
        if (notrans) {
            // Form X := alpha*inv(A)*B
            if (upper) {
                for (int j = 0; j < n; j++) {
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = alpha * b[i + j * ldb];
                        }
                    }
                    for (int k = m - 1; k >= 0; k--) {
                        if (b[k + j * ldb] != zero) {
                            if (nounit) b[k + j * ldb] = b[k + j * ldb] / a[k + k * lda];
                            for (int i = 0; i < k; i++) {
                                b[i + j * ldb] -= b[k + j * ldb] * a[i + k * lda];
                            }
                        }
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = alpha * b[i + j * ldb];
                        }
                    }
                    for (int k = 0; k < m; k++) {
                        if (b[k + j * ldb] != zero) {
                            if (nounit) b[k + j * ldb] = b[k + j * ldb] / a[k + k * lda];
                            for (int i = k + 1; i < m; i++) {
                                b[i + j * ldb] -= b[k + j * ldb] * a[i + k * lda];
                            }
                        }
                    }
                }
            }
        } else {
            // Form X := alpha*inv(A^T)*B
            if (upper) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        double temp = alpha * b[i + j * ldb];
                        for (int k = 0; k < i; k++) {
                            temp -= a[k + i * lda] * b[k + j * ldb];
                        }
                        if (nounit) temp = temp / a[i + i * lda];
                        b[i + j * ldb] = temp;
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    for (int i = m - 1; i >= 0; i--) {
                        double temp = alpha * b[i + j * ldb];
                        for (int k = i + 1; k < m; k++) {
                            temp -= a[k + i * lda] * b[k + j * ldb];
                        }
                        if (nounit) temp = temp / a[i + i * lda];
                        b[i + j * ldb] = temp;
                    }
                }
            }
        }
    } else {
        if (notrans) {
            // Form X := alpha*B*inv(A)
            if (upper) {
                for (int j = 0; j < n; j++) {
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = alpha * b[i + j * ldb];
                        }
                    }
                    for (int k = 0; k < j; k++) {
                        if (a[k + j * lda] != zero) {
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] -= a[k + j * lda] * b[i + k * ldb];
                            }
                        }
                    }
                    if (nounit) {
                        double temp = one / a[j + j * lda];
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = temp * b[i + j * ldb];
                        }
                    }
                }
            } else {
                for (int j = n - 1; j >= 0; j--) {
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = alpha * b[i + j * ldb];
                        }
                    }
                    for (int k = j + 1; k < n; k++) {
                        if (a[k + j * lda] != zero) {
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] -= a[k + j * lda] * b[i + k * ldb];
                            }
                        }
                    }
                    if (nounit) {
                        double temp = one / a[j + j * lda];
                        for (int i = 0; i < m; i++) {
                            b[i + j * ldb] = temp * b[i + j * ldb];
                        }
                    }
                }
            }
        } else {
            // Form X := alpha*B*inv(A^T)
            if (upper) {
                for (int k = n - 1; k >= 0; k--) {
                    if (nounit) {
                        double temp = one / a[k + k * lda];
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = temp * b[i + k * ldb];
                        }
                    }
                    for (int j = 0; j < k; j++) {
                        if (a[j + k * lda] != zero) {
                            double temp = a[j + k * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] -= temp * b[i + k * ldb];
                            }
                        }
                    }
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = alpha * b[i + k * ldb];
                        }
                    }
                }
            } else {
                for (int k = 0; k < n; k++) {
                    if (nounit) {
                        double temp = one / a[k + k * lda];
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = temp * b[i + k * ldb];
                        }
                    }
                    for (int j = k + 1; j < n; j++) {
                        if (a[j + k * lda] != zero) {
                            double temp = a[j + k * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] -= temp * b[i + k * ldb];
                            }
                        }
                    }
                    if (alpha != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = alpha * b[i + k * ldb];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
