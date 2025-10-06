#include <algorithm>
#include <cmath>

extern "C" {

void dsbmv(int uplo, int n, int k, double alpha, 
           const double* a, int lda, const double* x, int incx, 
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

    if (uplo == 0) {  // Upper triangle stored
        int kplus1 = k + 1;
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = 0.0;
                int l = kplus1 - 1 - j;
                
                // Process the strict upper triangular part
                int i_start = std::max(0, j - k);
                for (int i = i_start; i < j; i++) {
                    y[i] += temp1 * a[(l + i) + j * lda];
                    temp2 += a[(l + i) + j * lda] * x[i];
                }
                
                // Process the diagonal element
                y[j] += temp1 * a[kplus1 - 1 + j * lda] + alpha * temp2;
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = 0.0;
                int ix = kx;
                int iy = ky;
                int l = kplus1 - 1 - j;
                
                // Process the strict upper triangular part
                int i_start = std::max(0, j - k);
                for (int i = i_start; i < j; i++) {
                    y[iy] += temp1 * a[(l + i) + j * lda];
                    temp2 += a[(l + i) + j * lda] * x[ix];
                    ix += incx;
                    iy += incy;
                }
                
                // Process the diagonal element
                y[jy] += temp1 * a[kplus1 - 1 + j * lda] + alpha * temp2;
                jx += incx;
                jy += incy;
                if (j >= k) {
                    kx += incx;
                    ky += incy;
                }
            }
        }
    } else {  // Lower triangle stored
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[j];
                double temp2 = 0.0;
                
                // Process the diagonal element
                y[j] += temp1 * a[0 + j * lda];
                
                // Process the strict lower triangular part
                int l = 1 - j;
                int i_end = std::min(n - 1, j + k);
                for (int i = j + 1; i <= i_end; i++) {
                    y[i] += temp1 * a[(l + i) + j * lda];
                    temp2 += a[(l + i) + j * lda] * x[i];
                }
                
                y[j] += alpha * temp2;
            }
        } else {
            int jx = kx;
            int jy = ky;
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * x[jx];
                double temp2 = 0.0;
                
                // Process the diagonal element
                y[jy] += temp1 * a[0 + j * lda];
                
                // Process the strict lower triangular part
                int l = 1 - j;
                int ix = jx;
                int iy = jy;
                int i_end = std::min(n - 1, j + k);
                for (int i = j + 1; i <= i_end; i++) {
                    ix += incx;
                    iy += incy;
                    y[iy] += temp1 * a[(l + i) + j * lda];
                    temp2 += a[(l + i) + j * lda] * x[ix];
                }
                
                y[jy] += alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    }
}

} // extern "C"
