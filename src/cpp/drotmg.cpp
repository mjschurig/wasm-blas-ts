/**
 * DROTMG - Double precision modified Givens rotation generation
 * 
 * Constructs the modified Givens transformation matrix H which zeros
 * the second component of the 2-vector (sqrt(dd1)*dx1, sqrt(dd2)*dy1)^T
 * 
 * This is a C++ implementation of the BLAS Level 1 DROTMG routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param dd1    Input/output diagonal element
 * @param dd2    Input/output diagonal element  
 * @param dx1    Input/output vector element
 * @param dy1    Input vector element
 * @param param  Output parameter array [flag, h11, h21, h12, h22]
 */

#include <cmath>

extern "C" {

void drotmg(double* dd1, double* dd2, double* dx1, double dy1, double* param) {
    const double zero = 0.0;
    const double one = 1.0;
    const double two = 2.0;
    const double gam = 4096.0;
    const double gamsq = 16777216.0;
    const double rgamsq = 5.9604645e-8;
    
    double dflag, dh11, dh12, dh21, dh22;
    double dp1, dp2, dq1, dq2, dtemp, du;
    
    if (*dd1 < zero) {
        // Go zero-H-D-and-DX1
        dflag = -one;
        dh11 = zero;
        dh12 = zero;
        dh21 = zero;
        dh22 = zero;
        
        *dd1 = zero;
        *dd2 = zero;
        *dx1 = zero;
    } else {
        // Case DD1 non-negative
        dp2 = *dd2 * dy1;
        if (dp2 == zero) {
            dflag = -two;
            param[0] = dflag;
            return;
        }
        
        // Regular case
        dp1 = *dd1 * *dx1;
        dq2 = dp2 * dy1;
        dq1 = dp1 * *dx1;
        
        if (std::abs(dq1) > std::abs(dq2)) {
            dh21 = -dy1 / *dx1;
            dh12 = dp2 / dp1;
            
            du = one - dh12 * dh21;
            
            if (du > zero) {
                dflag = zero;
                *dd1 = *dd1 / du;
                *dd2 = *dd2 / du;
                *dx1 = *dx1 * du;
            } else {
                // Safety path for edge cases with rounding errors
                dflag = -one;
                dh11 = zero;
                dh12 = zero;
                dh21 = zero;
                dh22 = zero;
                
                *dd1 = zero;
                *dd2 = zero;
                *dx1 = zero;
            }
        } else {
            if (dq2 < zero) {
                // Go zero-H-D-and-DX1
                dflag = -one;
                dh11 = zero;
                dh12 = zero;
                dh21 = zero;
                dh22 = zero;
                
                *dd1 = zero;
                *dd2 = zero;
                *dx1 = zero;
            } else {
                dflag = one;
                dh11 = dp1 / dp2;
                dh22 = *dx1 / dy1;
                du = one + dh11 * dh22;
                dtemp = *dd2 / du;
                *dd2 = *dd1 / du;
                *dd1 = dtemp;
                *dx1 = dy1 * du;
            }
        }
        
        // Procedure: Scale-check
        if (*dd1 != zero) {
            while ((*dd1 <= rgamsq) || (*dd1 >= gamsq)) {
                if (dflag == zero) {
                    dh11 = one;
                    dh22 = one;
                    dflag = -one;
                } else {
                    dh21 = -one;
                    dh12 = one;
                    dflag = -one;
                }
                
                if (*dd1 <= rgamsq) {
                    *dd1 = *dd1 * gamsq;
                    *dx1 = *dx1 / gam;
                    dh11 = dh11 / gam;
                    dh12 = dh12 / gam;
                } else {
                    *dd1 = *dd1 / gamsq;
                    *dx1 = *dx1 * gam;
                    dh11 = dh11 * gam;
                    dh12 = dh12 * gam;
                }
            }
        }
        
        if (*dd2 != zero) {
            while ((std::abs(*dd2) <= rgamsq) || (std::abs(*dd2) >= gamsq)) {
                if (dflag == zero) {
                    dh11 = one;
                    dh22 = one;
                    dflag = -one;
                } else {
                    dh21 = -one;
                    dh12 = one;
                    dflag = -one;
                }
                
                if (std::abs(*dd2) <= rgamsq) {
                    *dd2 = *dd2 * gamsq;
                    dh21 = dh21 / gam;
                    dh22 = dh22 / gam;
                } else {
                    *dd2 = *dd2 / gamsq;
                    dh21 = dh21 * gam;
                    dh22 = dh22 * gam;
                }
            }
        }
    }
    
    // Set parameter array based on flag
    if (dflag < zero) {
        param[1] = dh11;
        param[2] = dh21;
        param[3] = dh12;
        param[4] = dh22;
    } else if (dflag == zero) {
        param[2] = dh21;
        param[3] = dh12;
    } else {
        param[1] = dh11;
        param[4] = dh22;
    }
    
    param[0] = dflag;
}

} // extern "C"
