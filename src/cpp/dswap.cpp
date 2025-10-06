/**
 * DSWAP - Double precision vector swap
 * 
 * Computes: swap x and y
 * 
 * This is a C++ implementation of the BLAS Level 1 DSWAP routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param x      Input/output vector x
 * @param incx   Storage spacing between elements of x
 * @param y      Input/output vector y
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void dswap(int n, double* x, int incx, double* y, int incy) {
    // Quick return if possible
    if (n <= 0) return;
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        // Clean-up loop - handle remainder when n is not divisible by 3
        int m = n % 3;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                double dtemp = x[i];
                x[i] = y[i];
                y[i] = dtemp;
            }
            if (n < 3) return;
        }
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 3) {
            double dtemp = x[i];
            x[i] = y[i];
            y[i] = dtemp;
            
            dtemp = x[i + 1];
            x[i + 1] = y[i + 1];
            y[i + 1] = dtemp;
            
            dtemp = x[i + 2];
            x[i + 2] = y[i + 2];
            y[i + 2] = dtemp;
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            double dtemp = x[ix];
            x[ix] = y[iy];
            y[iy] = dtemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

} // extern "C"
