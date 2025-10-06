/**
 * DROTG - Double precision Givens rotation generation
 * 
 * Constructs a plane rotation that eliminates the second component of a vector
 * 
 * This is a C++ implementation of the BLAS Level 1 DROTG routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param a      Input scalar a, overwritten with r
 * @param b      Input scalar b, overwritten with z  
 * @param c      Output cosine of rotation angle
 * @param s      Output sine of rotation angle
 */

#include <cmath>
#include <limits>
#include <algorithm>

extern "C" {

void drotg(double* a, double* b, double* c, double* s) {
    const double zero = 0.0;
    const double one = 1.0;
    
    // Scaling constants for safe computation
    const double safmin = std::pow(2.0, std::max(
        std::numeric_limits<double>::min_exponent - 1,
        1 - std::numeric_limits<double>::max_exponent
    ));
    const double safmax = std::pow(2.0, std::max(
        1 - std::numeric_limits<double>::min_exponent,
        std::numeric_limits<double>::max_exponent - 1
    ));
    
    double anorm = std::abs(*a);
    double bnorm = std::abs(*b);
    
    if (bnorm == zero) {
        *c = one;
        *s = zero;
        *b = zero;
    } else if (anorm == zero) {
        *c = zero;
        *s = one;
        *a = *b;
        *b = one;
    } else {
        double scl = std::min(safmax, std::max(safmin, std::max(anorm, bnorm)));
        double sigma;
        
        if (anorm > bnorm) {
            sigma = (*a >= zero) ? one : -one;
        } else {
            sigma = (*b >= zero) ? one : -one;
        }
        
        double r = sigma * (scl * std::sqrt((*a / scl) * (*a / scl) + (*b / scl) * (*b / scl)));
        *c = *a / r;
        *s = *b / r;
        
        double z;
        if (anorm > bnorm) {
            z = *s;
        } else if (*c != zero) {
            z = one / *c;
        } else {
            z = one;
        }
        
        *a = r;
        *b = z;
    }
}

} // extern "C"
