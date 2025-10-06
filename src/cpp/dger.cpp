/**
 * DGER - Double precision general rank-1 update
 * 
 * Computes: A := alpha * x * y^T + A
 * 
 * This is a C++ implementation of the BLAS Level 2 DGER routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param m      Number of rows of matrix A
 * @param n      Number of columns of matrix A
 * @param alpha  Scalar multiplier
 * @param x      Input vector x (m elements)
 * @param incx   Storage spacing between elements of x
 * @param y      Input vector y (n elements)
 * @param incy   Storage spacing between elements of y
 * @param a      Input/output matrix A (m x n, column-major)
 * @param lda    Leading dimension of A
 */

extern "C" {

void dger(int m, int n, double alpha, const double* x, int incx,
          const double* y, int incy, double* a, int lda) {
    
    const double zero = 0.0;
    
    // Quick return if possible
    if (m == 0 || n == 0 || alpha == zero) return;
    
    // Set up the start points in X and Y
    int kx = 0, ky = 0;
    if (incx < 0) kx = (-m + 1) * incx;
    if (incy < 0) ky = (-n + 1) * incy;
    
    // Start the operations
    if (incy == 1) {
        // Form A := alpha*x*y^T + A when y increment is 1
        int jx = kx;
        if (incx == 1) {
            // Both increments equal to 1
            for (int j = 0; j < n; j++) {
                if (y[j] != zero) {
                    double temp = alpha * y[j];
                    for (int i = 0; i < m; i++) {
                        a[i + j * lda] += temp * x[i];
                    }
                }
            }
        } else {
            // incx != 1
            for (int j = 0; j < n; j++) {
                if (y[j] != zero) {
                    double temp = alpha * y[j];
                    int ix = kx;
                    for (int i = 0; i < m; i++) {
                        a[i + j * lda] += temp * x[ix];
                        ix += incx;
                    }
                }
            }
        }
    } else {
        // Form A := alpha*x*y^T + A when y increment is not 1
        int jy = ky;
        if (incx == 1) {
            // incx == 1, incy != 1
            for (int j = 0; j < n; j++) {
                if (y[jy] != zero) {
                    double temp = alpha * y[jy];
                    for (int i = 0; i < m; i++) {
                        a[i + j * lda] += temp * x[i];
                    }
                }
                jy += incy;
            }
        } else {
            // Both increments not equal to 1
            for (int j = 0; j < n; j++) {
                if (y[jy] != zero) {
                    double temp = alpha * y[jy];
                    int ix = kx;
                    for (int i = 0; i < m; i++) {
                        a[i + j * lda] += temp * x[ix];
                        ix += incx;
                    }
                }
                jy += incy;
            }
        }
    }
}

} // extern "C"
