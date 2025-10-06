/**
 * DTRMM - Double precision triangular matrix-matrix multiplication
 * 
 * Computes: B := alpha*op(A)*B  or  B := alpha*B*op(A)
 * where op(A) = A or A^T and A is triangular
 * 
 * This is a C++ implementation of the BLAS Level 3 DTRMM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param side   'L': B := alpha*op(A)*B, 'R': B := alpha*B*op(A)
 * @param uplo   'U': upper triangular, 'L': lower triangular
 * @param transa 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param diag   'U': unit triangular, 'N': non-unit triangular
 * @param m      Number of rows of matrix B
 * @param n      Number of columns of matrix B
 * @param alpha  Scalar multiplier
 * @param a      Triangular matrix A
 * @param lda    Leading dimension of A
 * @param b      Input/output matrix B
 * @param ldb    Leading dimension of B
 */

extern "C" {

void dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha,
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
            // Form B := alpha*A*B
            if (upper) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < m; k++) {
                        if (b[k + j * ldb] != zero) {
                            double temp = alpha * b[k + j * ldb];
                            for (int i = 0; i < k; i++) {
                                b[i + j * ldb] += temp * a[i + k * lda];
                            }
                            if (nounit) temp = temp * a[k + k * lda];
                            b[k + j * ldb] = temp;
                        }
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    for (int k = m - 1; k >= 0; k--) {
                        if (b[k + j * ldb] != zero) {
                            double temp = alpha * b[k + j * ldb];
                            b[k + j * ldb] = temp;
                            if (nounit) b[k + j * ldb] = b[k + j * ldb] * a[k + k * lda];
                            for (int i = k + 1; i < m; i++) {
                                b[i + j * ldb] += temp * a[i + k * lda];
                            }
                        }
                    }
                }
            }
        } else {
            // Form B := alpha*A^T*B
            if (upper) {
                for (int j = 0; j < n; j++) {
                    for (int i = m - 1; i >= 0; i--) {
                        double temp = b[i + j * ldb];
                        if (nounit) temp = temp * a[i + i * lda];
                        for (int k = 0; k < i; k++) {
                            temp += a[k + i * lda] * b[k + j * ldb];
                        }
                        b[i + j * ldb] = alpha * temp;
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        double temp = b[i + j * ldb];
                        if (nounit) temp = temp * a[i + i * lda];
                        for (int k = i + 1; k < m; k++) {
                            temp += a[k + i * lda] * b[k + j * ldb];
                        }
                        b[i + j * ldb] = alpha * temp;
                    }
                }
            }
        }
    } else {
        if (notrans) {
            // Form B := alpha*B*A
            if (upper) {
                for (int j = n - 1; j >= 0; j--) {
                    double temp = alpha;
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = 0; i < m; i++) {
                        b[i + j * ldb] = temp * b[i + j * ldb];
                    }
                    for (int k = 0; k < j; k++) {
                        if (a[k + j * lda] != zero) {
                            temp = alpha * a[k + j * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] += temp * b[i + k * ldb];
                            }
                        }
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    double temp = alpha;
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = 0; i < m; i++) {
                        b[i + j * ldb] = temp * b[i + j * ldb];
                    }
                    for (int k = j + 1; k < n; k++) {
                        if (a[k + j * lda] != zero) {
                            temp = alpha * a[k + j * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] += temp * b[i + k * ldb];
                            }
                        }
                    }
                }
            }
        } else {
            // Form B := alpha*B*A^T
            if (upper) {
                for (int k = 0; k < n; k++) {
                    for (int j = 0; j < k; j++) {
                        if (a[j + k * lda] != zero) {
                            double temp = alpha * a[j + k * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] += temp * b[i + k * ldb];
                            }
                        }
                    }
                    double temp = alpha;
                    if (nounit) temp = temp * a[k + k * lda];
                    if (temp != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = temp * b[i + k * ldb];
                        }
                    }
                }
            } else {
                for (int k = n - 1; k >= 0; k--) {
                    for (int j = k + 1; j < n; j++) {
                        if (a[j + k * lda] != zero) {
                            double temp = alpha * a[j + k * lda];
                            for (int i = 0; i < m; i++) {
                                b[i + j * ldb] += temp * b[i + k * ldb];
                            }
                        }
                    }
                    double temp = alpha;
                    if (nounit) temp = temp * a[k + k * lda];
                    if (temp != one) {
                        for (int i = 0; i < m; i++) {
                            b[i + k * ldb] = temp * b[i + k * ldb];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
