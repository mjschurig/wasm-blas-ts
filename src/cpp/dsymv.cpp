/**
 * DSYMV - Double precision symmetric matrix-vector multiplication
 * 
 * Computes: y := alpha * A * x + beta * y
 * where A is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 2 DSYMV routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part
 * @param n      Order of the matrix A
 * @param alpha  Scalar multiplier for A*x
 * @param a      Symmetric matrix A (n x n, column-major)
 * @param lda    Leading dimension of A
 * @param x      Input vector x (n elements)
 * @param incx   Storage spacing between elements of x
 * @param beta   Scalar multiplier for y
 * @param y      Input/output vector y (n elements)
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void dsymv(char uplo, int n, double alpha, const double* a, int lda,
           const double* x, int incx, double beta, double* y, int incy) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    // Quick return if possible
    if (n == 0 || (alpha == zero && beta == one)) return;
    
    // Set up the start points in X and Y
    int kx = 0, ky = 0;
    if (incx < 0) kx = (-n + 1) * incx;
    if (incy < 0) ky = (-n + 1) * incy;
    
    // Start the operations. In this version the elements of A are accessed sequentially
    
    // First form y := beta*y
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int i = 0; i < n; i++) {
                    y[i] = zero;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            int iy = ky;
            if (beta == zero) {
                for (int i = 0; i < n; i++) {
                    y[iy] = zero;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }
    
    if (alpha == zero) return;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    
    if (incx == 1 && incy == 1) {
        // Both increments equal to 1
        if (upper) {
            // Form y when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = zero;
                for (int i = 0; i < j; i++) {
                    y[i] += temp1 * a[i + j * lda];
                    temp2 += a[i + j * lda] * x[i];
                }
                y[j] += temp1 * a[j + j * lda] + alpha * temp2;
            }
        } else {
            // Form y when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = zero;
                y[j] += temp1 * a[j + j * lda];
                for (int i = j + 1; i < n; i++) {
                    y[i] += temp1 * a[i + j * lda];
                    temp2 += a[i + j * lda] * x[i];
                }
                y[j] += alpha * temp2;
            }
        }
    } else {
        // General case with arbitrary increments
        int jx = kx;
        int jy = ky;
        
        if (upper) {
            // Form y when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = zero;
                int ix = kx;
                int iy = ky;
                for (int i = 0; i < j; i++) {
                    y[iy] += temp1 * a[i + j * lda];
                    temp2 += a[i + j * lda] * x[ix];
                    ix += incx;
                    iy += incy;
                }
                y[jy] += temp1 * a[j + j * lda] + alpha * temp2;
                jx += incx;
                jy += incy;
            }
        } else {
            // Form y when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = zero;
                y[jy] += temp1 * a[j + j * lda];
                int ix = jx;
                int iy = jy;
                ix += incx;
                iy += incy;
                for (int i = j + 1; i < n; i++) {
                    y[iy] += temp1 * a[i + j * lda];
                    temp2 += a[i + j * lda] * x[ix];
                    ix += incx;
                    iy += incy;
                }
                y[jy] += alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    }
}

} // extern "C"
