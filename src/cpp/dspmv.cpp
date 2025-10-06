#include <algorithm>
#include <cmath>

extern "C" {

void dspmv(int uplo, int n, double alpha, 
           const double* ap, const double* x, int incx, 
           double beta, double* y, int incy) {
    // Quick return if possible
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Set up the start points in X and Y
    int kx = 0, ky = 0;
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

    // First form y := beta*y
    if (beta != 1.0) {
        if (incy == 1) {
            if (beta == 0.0) {
                for (int i = 0; i < n; i++) {
                    y[i] = 0.0;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            int iy = ky;
            if (beta == 0.0) {
                for (int i = 0; i < n; i++) {
                    y[iy] = 0.0;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }

    if (alpha == 0.0) return;

    int kk = 0;  // Index into packed array
    
    if (uplo == 0) {  // Upper triangle stored
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = 0.0;
                int k = kk;
                
                // Process the strict upper triangular part
                for (int i = 0; i < j; i++) {
                    y[i] += temp1 * ap[k];
                    temp2 += ap[k] * x[i];
                    k++;
                }
                
                // Process the diagonal element
                y[j] += temp1 * ap[kk + j] + alpha * temp2;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = 0.0;
                int ix = kx;
                int iy = ky;
                
                // Process the strict upper triangular part
                for (int k = kk; k < kk + j; k++) {
                    y[iy] += temp1 * ap[k];
                    temp2 += ap[k] * x[ix];
                    ix += incx;
                    iy += incy;
                }
                
                // Process the diagonal element
                y[jy] += temp1 * ap[kk + j] + alpha * temp2;
                jx += incx;
                jy += incy;
                kk += j + 1;
            }
        }
    } else {  // Lower triangle stored
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = 0.0;
                
                // Process the diagonal element
                y[j] += temp1 * ap[kk];
                int k = kk + 1;
                
                // Process the strict lower triangular part
                for (int i = j + 1; i < n; i++) {
                    y[i] += temp1 * ap[k];
                    temp2 += ap[k] * x[i];
                    k++;
                }
                
                y[j] += alpha * temp2;
                kk += (n - j);
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = 0.0;
                
                // Process the diagonal element
                y[jy] += temp1 * ap[kk];
                int ix = jx;
                int iy = jy;
                
                // Process the strict lower triangular part
                for (int k = kk + 1; k < kk + n - j; k++) {
                    ix += incx;
                    iy += incy;
                    y[iy] += temp1 * ap[k];
                    temp2 += ap[k] * x[ix];
                }
                
                y[jy] += alpha * temp2;
                jx += incx;
                jy += incy;
                kk += (n - j);
            }
        }
    }
}

} // extern "C"
