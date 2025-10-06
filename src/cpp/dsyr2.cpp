/**
 * DSYR2 - Double precision symmetric rank-2 update
 * 
 * Computes: A := alpha * x * y^T + alpha * y * x^T + A
 * where A is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 2 DSYR2 routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part
 * @param n      Order of the matrix A
 * @param alpha  Scalar multiplier
 * @param x      Input vector x (n elements)
 * @param incx   Storage spacing between elements of x
 * @param y      Input vector y (n elements)
 * @param incy   Storage spacing between elements of y
 * @param a      Input/output symmetric matrix A (n x n, column-major) 
 * @param lda    Leading dimension of A
 */

extern "C" {

void dsyr2(char uplo, int n, double alpha, const double* x, int incx,
           const double* y, int incy, double* a, int lda) {
    
    const double zero = 0.0;
    
    // Quick return if possible
    if (n == 0 || alpha == zero) return;
    
    // Set up the start points in X and Y
    int kx = 0, ky = 0;
    if (incx < 0) kx = (-n + 1) * incx;
    if (incy < 0) ky = (-n + 1) * incy;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    
    if (incx == 1 && incy == 1) {
        // Form A when both increments are 1
        if (upper) {
            // Form A when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                if (x[j] != zero || y[j] != zero) {
                    double temp1 = alpha * y[j];
                    double temp2 = alpha * x[j];
                    for (int i = 0; i <= j; i++) {
                        a[i + j * lda] += x[i] * temp1 + y[i] * temp2;
                    }
                }
            }
        } else {
            // Form A when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                if (x[j] != zero || y[j] != zero) {
                    double temp1 = alpha * y[j];
                    double temp2 = alpha * x[j];
                    for (int i = j; i < n; i++) {
                        a[i + j * lda] += x[i] * temp1 + y[i] * temp2;
                    }
                }
            }
        }
    } else {
        // Form A when increments are not both 1
        int jx = kx;
        int jy = ky;
        if (upper) {
            // Form A when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                if (x[jx] != zero || y[jy] != zero) {
                    double temp1 = alpha * y[jy];
                    double temp2 = alpha * x[jx];
                    int ix = kx;
                    int iy = ky;
                    for (int i = 0; i <= j; i++) {
                        a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
                        ix += incx;
                        iy += incy;
                    }
                }
                jx += incx;
                jy += incy;
            }
        } else {
            // Form A when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                if (x[jx] != zero || y[jy] != zero) {
                    double temp1 = alpha * y[jy];
                    double temp2 = alpha * x[jx];
                    int ix = jx;
                    int iy = jy;
                    for (int i = j; i < n; i++) {
                        a[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
                        ix += incx;
                        iy += incy;
                    }
                }
                jx += incx;
                jy += incy;
            }
        }
    }
}

} // extern "C"
