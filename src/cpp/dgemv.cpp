/**
 * DGEMV - Double precision general matrix-vector multiplication
 * 
 * Computes: y = alpha * A * x + beta * y  or  y = alpha * A^T * x + beta * y
 * 
 * This is a C++ implementation of the BLAS Level 2 DGEMV routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param trans  0: y = alpha*A*x + beta*y, 1/2: y = alpha*A^T*x + beta*y
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

void dgemv(int trans, int m, int n, double alpha, const double* a, int lda,
           const double* x, int incx, double beta, double* y, int incy) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    // Test the input parameters
    bool notran = (trans == 0);
    
    // Quick return if possible
    if (m == 0 || n == 0 || (alpha == zero && beta == one)) return;
    
    // Set LENX and LENY, the lengths of the vectors x and y, and set
    // up the start points in X and Y.
    int lenx, leny;
    if (notran) {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }

    int kx = 0, ky = 0;
    if (incx < 0) kx = (1 - lenx) * incx;
    if (incy < 0) ky = (1 - leny) * incy;
    
    // Start the operations. In this version the elements of A are
    // accessed sequentially with one pass through A.
    
    // First form y := beta*y.
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int i = 0; i < leny; i++) {
                    y[i] = zero;
                }
            } else {
                for (int i = 0; i < leny; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            int iy = ky;
            if (beta == zero) {
                for (int i = 0; i < leny; i++) {
                    y[iy] = zero;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < leny; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }
    
    if (alpha == zero) return;
    
    if (notran) {
        // Form y := alpha*A*x + y.
        int jx = kx;
        if (incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp = alpha * x[jx];
                for (int i = 0; i < m; i++) {
                    y[i] = y[i] + temp * a[i + j * lda];
                }
                jx += incx;
            }
        } else {
            for (int j = 0; j < n; j++) {
                double temp = alpha * x[jx];
                int iy = ky;
                for (int i = 0; i < m; i++) {
                    y[iy] = y[iy] + temp * a[i + j * lda];
                    iy += incy;
                }
                jx += incx;
            }
        }
    } else {
        // Form y := alpha*A^T*x + y.
        int jy = ky;
        if (incx == 1) {
            for (int j = 0; j < n; j++) {
                double temp = zero;
                for (int i = 0; i < m; i++) {
                    temp += a[i + j * lda] * x[i];
                }
                y[jy] = y[jy] + alpha * temp;
                jy += incy;
            }
        } else {
            for (int j = 0; j < n; j++) {
                double temp = zero;
                int ix = kx;
                for (int i = 0; i < m; i++) {
                    temp += a[i + j * lda] * x[ix];
                    ix += incx;
                }
                y[jy] = y[jy] + alpha * temp;
                jy += incy;
            }
        }
    }
}

} // extern "C"