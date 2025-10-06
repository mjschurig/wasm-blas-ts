#include <algorithm>
#include <cmath>

extern "C" {

void dspr2(int uplo, int n, double alpha, 
           const double* x, int incx, const double* y, int incy, double* ap) {
    // Quick return if possible
    if (n == 0 || alpha == 0.0) return;

    // Set up the start points in X and Y if the increments are not both unity
    int kx = 0, ky = 0;
    bool need_offsets = (incx != 1) || (incy != 1);
    
    if (need_offsets) {
        if (incx > 0) {
            kx = 0;
        } else {
            kx = -(n - 1) * incx;
        }
        if (incy > 0) {
            ky = 0;
        } else {
            ky = -(n - 1) * incy;
        }
    }

    // Start the operations. In this version the elements of the array AP
    // are accessed sequentially with one pass through AP.
    int kk = 0;
    
    if (uplo == 0) {  // Upper triangle stored in AP
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                if (x[j] != 0.0 || y[j] != 0.0) {
                    double temp1 = alpha * y[j];
                    double temp2 = alpha * x[j];
                    int k = kk;
                    for (int i = 0; i <= j; i++) {
                        ap[k] += x[i] * temp1 + y[i] * temp2;
                        k++;
                    }
                }
                kk += j + 1;
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                if (x[jx] != 0.0 || y[jy] != 0.0) {
                    double temp1 = alpha * y[jy];
                    double temp2 = alpha * x[jx];
                    int ix = kx;
                    int iy = ky;
                    for (int k = kk; k < kk + j + 1; k++) {
                        ap[k] += x[ix] * temp1 + y[iy] * temp2;
                        ix += incx;
                        iy += incy;
                    }
                }
                jx += incx;
                jy += incy;
                kk += j + 1;
            }
        }
    } else {  // Lower triangle stored in AP
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                if (x[j] != 0.0 || y[j] != 0.0) {
                    double temp1 = alpha * y[j];
                    double temp2 = alpha * x[j];
                    int k = kk;
                    for (int i = j; i < n; i++) {
                        ap[k] += x[i] * temp1 + y[i] * temp2;
                        k++;
                    }
                }
                kk += n - j;
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                if (x[jx] != 0.0 || y[jy] != 0.0) {
                    double temp1 = alpha * y[jy];
                    double temp2 = alpha * x[jx];
                    int ix = jx;
                    int iy = jy;
                    for (int k = kk; k < kk + n - j; k++) {
                        ap[k] += x[ix] * temp1 + y[iy] * temp2;
                        ix += incx;
                        iy += incy;
                    }
                }
                jx += incx;
                jy += incy;
                kk += n - j;
            }
        }
    }
}

} // extern "C"
