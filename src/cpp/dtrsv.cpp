/**
 * DTRSV - Double precision triangular solve
 * 
 * Solves: A*x = b  or  A^T*x = b
 * where A is a triangular matrix and b is overwritten by x
 * 
 * This is a C++ implementation of the BLAS Level 2 DTRSV routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': upper triangular, 'L': lower triangular
 * @param trans  'N': A*x = b, 'T'/'C': A^T*x = b
 * @param diag   'U': unit triangular, 'N': non-unit triangular
 * @param n      Order of the matrix A
 * @param a      Triangular matrix A (n x n, column-major)
 * @param lda    Leading dimension of A
 * @param x      Input/output vector (b on input, x on output)
 * @param incx   Storage spacing between elements of x
 */

extern "C" {

void dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda,
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
        // Form x := inv(A)*x
        if (upper) {
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    if (x[j] != zero) {
                        if (nounit) x[j] = x[j] / a[j + j * lda];
                        double temp = x[j];
                        for (int i = j - 1; i >= 0; i--) {
                            x[i] -= temp * a[i + j * lda];
                        }
                    }
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] = x[jx] / a[j + j * lda];
                        double temp = x[jx];
                        int ix = jx;
                        for (int i = j - 1; i >= 0; i--) {
                            ix -= incx;
                            x[ix] -= temp * a[i + j * lda];
                        }
                    }
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    if (x[j] != zero) {
                        if (nounit) x[j] = x[j] / a[j + j * lda];
                        double temp = x[j];
                        for (int i = j + 1; i < n; i++) {
                            x[i] -= temp * a[i + j * lda];
                        }
                    }
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] = x[jx] / a[j + j * lda];
                        double temp = x[jx];
                        int ix = jx;
                        for (int i = j + 1; i < n; i++) {
                            ix += incx;
                            x[ix] -= temp * a[i + j * lda];
                        }
                    }
                    jx += incx;
                }
            }
        }
    } else {
        // Form x := inv(A^T)*x
        if (upper) {
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    double temp = x[j];
                    for (int i = 0; i < j; i++) {
                        temp -= a[i + j * lda] * x[i];
                    }
                    if (nounit) temp = temp / a[j + j * lda];
                    x[j] = temp;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    double temp = x[jx];
                    int ix = kx;
                    for (int i = 0; i < j; i++) {
                        temp -= a[i + j * lda] * x[ix];
                        ix += incx;
                    }
                    if (nounit) temp = temp / a[j + j * lda];
                    x[jx] = temp;
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[j];
                    for (int i = j + 1; i < n; i++) {
                        temp -= a[i + j * lda] * x[i];
                    }
                    if (nounit) temp = temp / a[j + j * lda];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[jx];
                    int ix = kx;
                    for (int i = j + 1; i < n; i++) {
                        temp -= a[i + j * lda] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) temp = temp / a[j + j * lda];
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        }
    }
}

} // extern "C"
