/**
 * DAXPBY - Double precision extended AXPY
 * 
 * Computes: y = alpha * x + beta * y
 * 
 * This is a C++ implementation of the BLAS Level 1 DAXPBY routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param alpha  Scalar multiplier for x
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @param beta   Scalar multiplier for y
 * @param y      Input/output vector y (result stored here)
 * @param incy   Storage spacing between elements of y
 */

extern "C" {

void daxpby(int n, double alpha, const double* x, int incx, 
            double beta, double* y, int incy) {
    // Quick return if possible
    if (n <= 0) return;
    
    // Special case: if alpha == 0 and beta != 0, just scale y
    if (alpha == 0.0 && beta != 0.0) {
        // Scale y by beta (equivalent to dscal)
        if (incy == 1) {
            for (int i = 0; i < n; i++) {
                y[i] = beta * y[i];
            }
        } else {
            int nincx = n * incy;
            for (int i = 0; i < nincx; i += incy) {
                y[i] = beta * y[i];
            }
        }
        return;
    }
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            y[i] = beta * y[i] + alpha * x[i];
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            y[iy] = beta * y[iy] + alpha * x[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

} // extern "C"
