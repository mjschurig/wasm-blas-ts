/**
 * DNRM2 - Double precision Euclidean norm
 * 
 * Computes: result = sqrt(x^T * x)
 * 
 * This is a C++ implementation of the BLAS Level 1 DNRM2 routine,
 * based on the reference BLAS implementation from netlib.org
 * Uses Blue's algorithm for safe scaling to avoid overflow/underflow
 * 
 * @param n      Number of elements in input vector
 * @param x      Input vector x
 * @param incx   Storage spacing between elements of x
 * @return       Euclidean norm of x
 */

#include <cmath>
#include <limits>
#include <algorithm>

extern "C" {

double dnrm2(int n, const double* x, int incx) {
    // Quick return if possible
    if (n <= 0) return 0.0;
    
    // Blue's scaling constants
    const double tsml = std::pow(2.0, std::ceil((std::numeric_limits<double>::min_exponent - 1) * 0.5));
    const double tbig = std::pow(2.0, std::floor((std::numeric_limits<double>::max_exponent - std::numeric_limits<double>::digits + 1) * 0.5));
    const double ssml = std::pow(2.0, -std::floor((std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits) * 0.5));
    const double sbig = std::pow(2.0, -std::ceil((std::numeric_limits<double>::max_exponent + std::numeric_limits<double>::digits - 1) * 0.5));
    
    double scl = 1.0;
    double sumsq = 0.0;
    
    // Compute the sum of squares in 3 accumulators:
    // abig -- sums of squares scaled down to avoid overflow
    // asml -- sums of squares scaled up to avoid underflow  
    // amed -- sums of squares that do not require scaling
    bool notbig = true;
    double asml = 0.0;
    double amed = 0.0;
    double abig = 0.0;
    
    int ix = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    
    for (int i = 0; i < n; i++) {
        double ax = std::abs(x[ix]);
        if (ax > tbig) {
            abig = abig + (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if (ax < tsml) {
            if (notbig) asml = asml + (ax * ssml) * (ax * ssml);
        } else {
            amed = amed + ax * ax;
        }
        ix = ix + incx;
    }
    
    // Combine abig and amed or amed and asml if more than one
    // accumulator was used.
    if (abig > 0.0) {
        // Combine abig and amed if abig > 0.
        if (amed > 0.0 || amed != amed) { // Check for NaN
            abig = abig + (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if (asml > 0.0) {
        // Combine amed and asml if asml > 0.
        if (amed > 0.0 || amed != amed) { // Check for NaN
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            double ymin, ymax;
            if (asml > amed) {
                ymin = amed;
                ymax = asml;
            } else {
                ymin = asml;
                ymax = amed;
            }
            scl = 1.0;
            sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        // Otherwise all values are mid-range
        scl = 1.0;
        sumsq = amed;
    }
    
    return scl * std::sqrt(sumsq);
}

} // extern "C"
