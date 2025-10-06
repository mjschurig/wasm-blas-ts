/**
 * DDOT - Double precision dot product
 * 
 * Computes: result = x^T * y
 * 
 * This is a C++ implementation of the BLAS Level 1 DDOT routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @param y      Input vector y
 * @param incy   Storage spacing between elements of y
 * @return       Dot product of x and y
 */

extern "C" {

double ddot(int n, const double* x, int incx, const double* y, int incy) {
    double dtemp = 0.0;
    
    // Quick return if possible
    if (n <= 0) return 0.0;
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        // Clean-up loop - handle remainder when n is not divisible by 5
        int m = n % 5;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                dtemp = dtemp + x[i] * y[i];
            }
            if (n < 5) return dtemp;
        }
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 5) {
            dtemp = dtemp + x[i] * y[i] + x[i + 1] * y[i + 1] +
                    x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            dtemp = dtemp + x[ix] * y[iy];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
    
    return dtemp;
}

} // extern "C"
