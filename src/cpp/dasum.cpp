/**
 * DASUM - Double precision sum of absolute values
 * 
 * Computes: result = sum(|x[i]|)
 * 
 * This is a C++ implementation of the BLAS Level 1 DASUM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param n      Number of elements in input vector
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @return       Sum of absolute values of elements in x
 */

#include <cmath>

extern "C" {

double dasum(int n, const double* x, int incx) {
    double dtemp = 0.0;
    
    // Quick return if possible
    if (n <= 0 || incx <= 0) return 0.0;
    
    if (incx == 1) {
        // Code for increment equal to 1
        // Clean-up loop - handle remainder when n is not divisible by 6
        int m = n % 6;
        if (m != 0) {
            for (int i = 0; i < m; i++) {
                dtemp = dtemp + std::abs(x[i]);
            }
            if (n < 6) return dtemp;
        }
        
        // Unrolled loop for better performance
        for (int i = m; i < n; i += 6) {
            dtemp = dtemp + std::abs(x[i]) + std::abs(x[i + 1]) +
                    std::abs(x[i + 2]) + std::abs(x[i + 3]) +
                    std::abs(x[i + 4]) + std::abs(x[i + 5]);
        }
    } else {
        // Code for increment not equal to 1
        int nincx = n * incx;
        for (int i = 0; i < nincx; i += incx) {
            dtemp = dtemp + std::abs(x[i]);
        }
    }
    
    return dtemp;
}

} // extern "C"
