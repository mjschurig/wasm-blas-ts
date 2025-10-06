/**
 * DSYR - Double precision symmetric rank-1 update
 * 
 * Computes: A := alpha * x * x^T + A
 * where A is a symmetric matrix
 * 
 * This is a C++ implementation of the BLAS Level 2 DSYR routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param uplo   'U': use upper triangular part, 'L': use lower triangular part
 * @param n      Order of the matrix A
 * @param alpha  Scalar multiplier
 * @param x      Input vector x (n elements)
 * @param incx   Storage spacing between elements of x
 * @param a      Input/output symmetric matrix A (n x n, column-major) 
 * @param lda    Leading dimension of A
 */

extern "C" {

void dsyr(char uplo, int n, double alpha, const double* x, int incx,
          double* a, int lda) {
    
    const double zero = 0.0;
    
    // Quick return if possible
    if (n == 0 || alpha == zero) return;
    
    // Set up the start point in X
    int kx = 0;
    if (incx < 0) kx = (-n + 1) * incx;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    
    if (incx == 1) {
        // Form A when x increment is 1
        if (upper) {
            // Form A when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                if (x[j] != zero) {
                    double temp = alpha * x[j];
                    for (int i = 0; i <= j; i++) {
                        a[i + j * lda] += x[i] * temp;
                    }
                }
            }
        } else {
            // Form A when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                if (x[j] != zero) {
                    double temp = alpha * x[j];
                    for (int i = j; i < n; i++) {
                        a[i + j * lda] += x[i] * temp;
                    }
                }
            }
        }
    } else {
        // Form A when x increment is not 1
        int jx = kx;
        if (upper) {
            // Form A when A is stored in upper triangle
            for (int j = 0; j < n; j++) {
                if (x[jx] != zero) {
                    double temp = alpha * x[jx];
                    int ix = kx;
                    for (int i = 0; i <= j; i++) {
                        a[i + j * lda] += x[ix] * temp;
                        ix += incx;
                    }
                }
                jx += incx;
            }
        } else {
            // Form A when A is stored in lower triangle
            for (int j = 0; j < n; j++) {
                if (x[jx] != zero) {
                    double temp = alpha * x[jx];
                    int ix = jx;
                    for (int i = j; i < n; i++) {
                        a[i + j * lda] += x[ix] * temp;
                        ix += incx;
                    }
                }
                jx += incx;
            }
        }
    }
}

} // extern "C"
