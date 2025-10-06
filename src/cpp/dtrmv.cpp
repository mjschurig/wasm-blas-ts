/**
 * DTRMV - Double precision triangular matrix-vector multiplication
 * 
 * Computes: x := A*x  or  x := A^T*x
 * where A is a triangular matrix
 * 
 * This is a C++ implementation of the BLAS Level 2 DTRMV routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': upper triangular, 'L': lower triangular
 * @param trans  'N': x := A*x, 'T'/'C': x := A^T*x
 * @param diag   'U': unit triangular, 'N': non-unit triangular
 * @param n      Order of the matrix A
 * @param a      Triangular matrix A (n x n, column-major)
 * @param lda    Leading dimension of A
 * @param x      Input/output vector x (n elements)
 * @param incx   Storage spacing between elements of x
 */

extern "C" {

void dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda,
           double* x, int incx) {
    
    const double zero = 0.0;
    
    // Quick return if possible
    if (n == 0) return;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    bool notrans = (trans == 'N' || trans == 'n');
    bool nounit = (diag == 'N' || diag == 'n');
    
    // Set up the start point in X
    int kx = 0;
    if (incx < 0) kx = (-n + 1) * incx;
    
    if (notrans) {
        // Form x := A*x
        if (upper) {
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    if (x[j] != zero) {
                        double temp = x[j];
                        for (int i = 0; i < j; i++) {
                            x[i] += temp * a[i + j * lda];
                        }
                        if (nounit) x[j] = temp * a[j + j * lda];
                    }
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    if (x[jx] != zero) {
                        double temp = x[jx];
                        int ix = kx;
                        for (int i = 0; i < j; i++) {
                            x[ix] += temp * a[i + j * lda];
                            ix += incx;
                        }
                        if (nounit) x[jx] = temp * a[j + j * lda];
                    }
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    if (x[j] != zero) {
                        double temp = x[j];
                        for (int i = n - 1; i > j; i--) {
                            x[i] += temp * a[i + j * lda];
                        }
                        if (nounit) x[j] = temp * a[j + j * lda];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    if (x[jx] != zero) {
                        double temp = x[jx];
                        int ix = kx;
                        for (int i = n - 1; i > j; i--) {
                            x[ix] += temp * a[i + j * lda];
                            ix -= incx;
                        }
                        if (nounit) x[jx] = temp * a[j + j * lda];
                    }
                    jx -= incx;
                }
            }
        }
    } else {
        // Form x := A^T*x
        if (upper) {
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[j];
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = j - 1; i >= 0; i--) {
                        temp += a[i + j * lda] * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[jx];
                    int ix = jx;
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = j - 1; i >= 0; i--) {
                        ix -= incx;
                        temp += a[i + j * lda] * x[ix];
                    }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    double temp = x[j];
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = j + 1; i < n; i++) {
                        temp += a[i + j * lda] * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    double temp = x[jx];
                    int ix = jx;
                    if (nounit) temp = temp * a[j + j * lda];
                    for (int i = j + 1; i < n; i++) {
                        ix += incx;
                        temp += a[i + j * lda] * x[ix];
                    }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

} // extern "C"
