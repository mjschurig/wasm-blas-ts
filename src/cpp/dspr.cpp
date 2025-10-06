#include <algorithm>
#include <cmath>

extern "C" {

void dspr(int uplo, int n, double alpha, 
          const double* x, int incx, double* ap) {
    // Quick return if possible
    if (n == 0 || alpha == 0.0) return;

    // Set the start point in X if the increment is not unity
    int kx = 0;
    if (incx <= 0) {
        kx = -(n - 1) * incx;
    } else if (incx != 1) {
        kx = 0;
    }

    // Start the operations. In this version the elements of the array AP
    // are accessed sequentially with one pass through AP.
    int kk = 0;
    
    if (uplo == 0) {  // Upper triangle stored in AP
        if (incx == 1) {
            for (int j = 0; j < n; j++) {
                if (x[j] != 0.0) {
                    double temp = alpha * x[j];
                    int k = kk;
                    for (int i = 0; i <= j; i++) {
                        ap[k] += x[i] * temp;
                        k++;
                    }
                }
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < n; j++) {
                if (x[jx] != 0.0) {
                    double temp = alpha * x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j + 1; k++) {
                        ap[k] += x[ix] * temp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        }
    } else {  // Lower triangle stored in AP
        if (incx == 1) {
            for (int j = 0; j < n; j++) {
                if (x[j] != 0.0) {
                    double temp = alpha * x[j];
                    int k = kk;
                    for (int i = j; i < n; i++) {
                        ap[k] += x[i] * temp;
                        k++;
                    }
                }
                kk += n - j;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < n; j++) {
                if (x[jx] != 0.0) {
                    double temp = alpha * x[jx];
                    int ix = jx;
                    for (int k = kk; k < kk + n - j; k++) {
                        ap[k] += x[ix] * temp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += n - j;
            }
        }
    }
}

} // extern "C"
