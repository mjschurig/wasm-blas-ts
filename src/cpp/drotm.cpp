/**
 * DROTM - Double precision modified Givens rotation
 * 
 * Applies a modified Givens transformation to vectors x and y
 * 
 * This is a C++ implementation of the BLAS Level 1 DROTM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n        Number of elements in input vectors
 * @param x        Input/output vector x
 * @param incx     Storage spacing between elements of x
 * @param y        Input/output vector y
 * @param incy     Storage spacing between elements of y
 * @param param    Parameter array with transformation matrix elements
 */

extern "C" {

void drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    const double zero = 0.0;
    const double two = 2.0;
    
    double dflag = param[0];
    
    // Quick return if possible
    if (n <= 0 || (dflag + two == zero)) return;
    
    if (incx == incy && incx > 0) {
        // Code for equal positive increments
        int nsteps = n * incx;
        
        if (dflag < zero) {
            // Full matrix
            double dh11 = param[1];
            double dh12 = param[3];
            double dh21 = param[2];
            double dh22 = param[4];
            
            for (int i = 0; i < nsteps; i += incx) {
                double w = x[i];
                double z = y[i];
                x[i] = w * dh11 + z * dh12;
                y[i] = w * dh21 + z * dh22;
            }
        } else if (dflag == zero) {
            // Identity with off-diagonal elements
            double dh12 = param[3];
            double dh21 = param[2];
            
            for (int i = 0; i < nsteps; i += incx) {
                double w = x[i];
                double z = y[i];
                x[i] = w + z * dh12;
                y[i] = w * dh21 + z;
            }
        } else {
            // Diagonal with special structure
            double dh11 = param[1];
            double dh22 = param[4];
            
            for (int i = 0; i < nsteps; i += incx) {
                double w = x[i];
                double z = y[i];
                x[i] = w * dh11 + z;
                y[i] = -w + dh22 * z;
            }
        }
    } else {
        // Code for unequal increments
        int kx = 0;
        int ky = 0;
        
        if (incx < 0) kx = (-n + 1) * incx;
        if (incy < 0) ky = (-n + 1) * incy;
        
        if (dflag < zero) {
            // Full matrix
            double dh11 = param[1];
            double dh12 = param[3];
            double dh21 = param[2];
            double dh22 = param[4];
            
            for (int i = 0; i < n; i++) {
                double w = x[kx];
                double z = y[ky];
                x[kx] = w * dh11 + z * dh12;
                y[ky] = w * dh21 + z * dh22;
                kx = kx + incx;
                ky = ky + incy;
            }
        } else if (dflag == zero) {
            // Identity with off-diagonal elements
            double dh12 = param[3];
            double dh21 = param[2];
            
            for (int i = 0; i < n; i++) {
                double w = x[kx];
                double z = y[ky];
                x[kx] = w + z * dh12;
                y[ky] = w * dh21 + z;
                kx = kx + incx;
                ky = ky + incy;
            }
        } else {
            // Diagonal with special structure
            double dh11 = param[1];
            double dh22 = param[4];
            
            for (int i = 0; i < n; i++) {
                double w = x[kx];
                double z = y[ky];
                x[kx] = w * dh11 + z;
                y[ky] = -w + dh22 * z;
                kx = kx + incx;
                ky = ky + incy;
            }
        }
    }
}

} // extern "C"
