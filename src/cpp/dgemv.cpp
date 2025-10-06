/**
 * DGEMV - Double precision general matrix-vector multiplication
 * 
 * Computes: y = alpha * A * x + beta * y  or  y = alpha * A^T * x + beta * y
 * 
 * This is a C++ implementation of the BLAS Level 2 DGEMV routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param trans  'N': y = alpha*A*x + beta*y, 'T'/'C': y = alpha*A^T*x + beta*y
 * @param m      Number of rows of matrix A
 * @param n      Number of columns of matrix A
 * @param alpha  Scalar multiplier for A*x or A^T*x 
 * @param a      Matrix A stored in column-major order
 * @param lda    Leading dimension of A (>= max(1,m))
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @param beta   Scalar multiplier for y
 * @param y      Input/output vector y
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void dgemv(char trans, int m, int n, double alpha, const double* a, int lda,
           const double* x, int incx, double beta, double* y, int incy) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    // Test the input parameters
    bool notran = (trans == 'N' || trans == 'n');
    
    // Quick return if possible
    if (m == 0 || n == 0 || (alpha == zero && beta == one)) return;
    
    // Set up the start points in X and Y
    int kx = 0, ky = 0;
    if (incx < 0) kx = (-n + 1) * incx;
    if (incy < 0) ky = (-m + 1) * incy;
    
    // Start the operations. In this version the elements of A are accessed sequentially
    if (notran) {
        // Form y := alpha*A*x + beta*y
        if (incx == 1) {
            if (incy == 1) {
                // Both increments equal to 1
                for (int j = 0; j < n; j++) {
                    if (x[j] != zero) {
                        double temp = alpha * x[j];
                        for (int i = 0; i < m; i++) {
                            y[i] = y[i] + temp * a[i + j * lda];
                        }
                    }
                }
                if (beta != one) {
                    if (beta == zero) {
                        for (int i = 0; i < m; i++) {
                            y[i] = zero;
                        }
                    } else {
                        for (int i = 0; i < m; i++) {
                            y[i] = beta * y[i];
                        }
                    }
                }
            } else {
                // incx = 1, incy != 1
                int iy = ky;
                for (int j = 0; j < n; j++) {
                    if (x[j] != zero) {
                        double temp = alpha * x[j];
                        int iy_temp = iy;
                        for (int i = 0; i < m; i++) {
                            y[iy_temp] = y[iy_temp] + temp * a[i + j * lda];
                            iy_temp += incy;
                        }
                    }
                }
                if (beta != one) {
                    iy = ky;
                    if (beta == zero) {
                        for (int i = 0; i < m; i++) {
                            y[iy] = zero;
                            iy += incy;
                        }
                    } else {
                        for (int i = 0; i < m; i++) {
                            y[iy] = beta * y[iy];
                            iy += incy;
                        }
                    }
                }
            }
        } else {
            // General case: incx != 1
            int jx = kx;
            if (incy == 1) {
                for (int j = 0; j < n; j++) {
                    if (x[jx] != zero) {
                        double temp = alpha * x[jx];
                        for (int i = 0; i < m; i++) {
                            y[i] = y[i] + temp * a[i + j * lda];
                        }
                    }
                    jx += incx;
                }
                if (beta != one) {
                    if (beta == zero) {
                        for (int i = 0; i < m; i++) {
                            y[i] = zero;
                        }
                    } else {
                        for (int i = 0; i < m; i++) {
                            y[i] = beta * y[i];
                        }
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    if (x[jx] != zero) {
                        double temp = alpha * x[jx];
                        int iy = ky;
                        for (int i = 0; i < m; i++) {
                            y[iy] = y[iy] + temp * a[i + j * lda];
                            iy += incy;
                        }
                    }
                    jx += incx;
                }
                if (beta != one) {
                    int iy = ky;
                    if (beta == zero) {
                        for (int i = 0; i < m; i++) {
                            y[iy] = zero;
                            iy += incy;
                        }
                    } else {
                        for (int i = 0; i < m; i++) {
                            y[iy] = beta * y[iy];
                            iy += incy;
                        }
                    }
                }
            }
        }
    } else {
        // Form y := alpha*A^T*x + beta*y
        int jy = ky;
        if (incx == 1) {
            if (incy == 1) {
                // Both increments equal to 1
                for (int j = 0; j < n; j++) {
                    double temp = zero;
                    for (int i = 0; i < m; i++) {
                        temp += a[i + j * lda] * x[i];
                    }
                    y[j] = alpha * temp + beta * y[j];
                }
            } else {
                // incx = 1, incy != 1
                for (int j = 0; j < n; j++) {
                    double temp = zero;
                    for (int i = 0; i < m; i++) {
                        temp += a[i + j * lda] * x[i];
                    }
                    y[jy] = alpha * temp + beta * y[jy];
                    jy += incy;
                }
            }
        } else {
            // General case: incx != 1
            if (incy == 1) {
                for (int j = 0; j < n; j++) {
                    double temp = zero;
                    int ix = kx;
                    for (int i = 0; i < m; i++) {
                        temp += a[i + j * lda] * x[ix];
                        ix += incx;
                    }
                    y[j] = alpha * temp + beta * y[j];
                }
            } else {
                for (int j = 0; j < n; j++) {
                    double temp = zero;
                    int ix = kx;
                    for (int i = 0; i < m; i++) {
                        temp += a[i + j * lda] * x[ix];
                        ix += incx;
                    }
                    y[jy] = alpha * temp + beta * y[jy];
                    jy += incy;
                }
            }
        }
    }
}

} // extern "C"
