/**
 * DROT - Double precision plane rotation
 * 
 * Computes: [x] = [c  s] [x]
 *           [y]   [-s c] [y]
 * 
 * This is a C++ implementation of the BLAS Level 1 DROT routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vectors
 * @param x      Input/output vector x
 * @param incx   Storage spacing between elements of x
 * @param y      Input/output vector y
 * @param incy   Storage spacing between elements of y
 * @param c      Cosine of the angle of rotation
 * @param s      Sine of the angle of rotation
 */

extern "C" {

void drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    // Quick return if possible
    if (n <= 0) return;
    
    // Code for both increments equal to 1
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double dtemp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = dtemp;
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        int ix = 0;
        int iy = 0;
        
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        
        for (int i = 0; i < n; i++) {
            double dtemp = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = dtemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

} // extern "C"
