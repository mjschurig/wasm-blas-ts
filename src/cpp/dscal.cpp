/**
 * DSCAL - Double precision vector scaling
 * 
 * Computes: x = alpha * x
 * 
 * This is a C++ implementation of the BLAS Level 1 DSCAL routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vector
 * @param alpha  Scalar multiplier
 * @param x      Input/output vector x (result stored here)
 * @param incx   Storage spacing between elements of x
 */

extern "C" {

void dscal(int n, double alpha, double* x, int incx) {
    // Quick return if possible
    if (n <= 0 || incx <= 0 || alpha == 1.0) return;
    
    if (incx == 1) {
        // Code for increment equal to 1
        // Clean-up loop - handle remainder when n is not divisible by 5
        int m = n % 5;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                x[i] = alpha * x[i];
            }
            if (n < 5) return;
        }
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 5) {
            x[i] = alpha * x[i];
            x[i + 1] = alpha * x[i + 1];
            x[i + 2] = alpha * x[i + 2];
            x[i + 3] = alpha * x[i + 3];
            x[i + 4] = alpha * x[i + 4];
        }
    } else {
        // Code for increment not equal to 1
        int nincx = n * incx;
        for (int i = 0; i < nincx; i += incx) {
            x[i] = alpha * x[i];
        }
    }
}

} // extern "C"
