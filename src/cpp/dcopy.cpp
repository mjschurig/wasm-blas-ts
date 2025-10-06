/**
 * DCOPY - Double precision vector copy
 * 
 * Computes: y = x
 * 
 * This is a C++ implementation of the BLAS Level 1 DCOPY routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @param y      Output vector y (result stored here)
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void dcopy(int n, const double* x, int incx, double* y, int incy) {
    // Quick return if possible
    if (n <= 0) return;
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        // Clean-up loop - handle remainder when n is not divisible by 7
        int m = n % 7;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                y[i] = x[i];
            }
            if (n < 7) return;
        }
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 7) {
            y[i] = x[i];
            y[i + 1] = x[i + 1];
            y[i + 2] = x[i + 2];
            y[i + 3] = x[i + 3];
            y[i + 4] = x[i + 4];
            y[i + 5] = x[i + 5];
            y[i + 6] = x[i + 6];
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            y[iy] = x[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

} // extern "C"
