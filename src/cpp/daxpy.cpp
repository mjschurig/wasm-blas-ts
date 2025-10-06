/**
 * DAXPY - Double precision A*X Plus Y
 * 
 * Computes: y = alpha * x + y
 * 
 * This is a C++ implementation of the BLAS Level 1 DAXPY routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param alpha  Scalar multiplier for x
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @param y      Input/output vector y (result stored here)
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    // Quick return if possible
    if (n <= 0) return;
    if (alpha == 0.0) return;
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        // Clean-up loop - handle remainder when n is not divisible by 4
        int m = n % 4;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                y[i] = y[i] + alpha * x[i];
            }
        }
        
        if (n < 4) return;
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 4) {
            y[i] = y[i] + alpha * x[i];
            y[i + 1] = y[i + 1] + alpha * x[i + 1];
            y[i + 2] = y[i + 2] + alpha * x[i + 2];
            y[i + 3] = y[i + 3] + alpha * x[i + 3];
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            y[iy] = y[iy] + alpha * x[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

} // extern "C"
