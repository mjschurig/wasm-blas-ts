#include <algorithm>
#include <cmath>

extern "C" {

void dgbmv(int trans, int m, int n, int kl, int ku, double alpha, 
           const double* a, int lda, const double* x, int incx, 
           double beta, double* y, int incy) {
    // Quick return if possible
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Set LENX and LENY
    int lenx, leny;
    if (trans == 0) {  // 'N'
        lenx = n;
        leny = m;
    } else {  // 'T' or 'C'
        lenx = m;
        leny = n;
    }

    // Set up the start points in X and Y
    int kx = 0, ky = 0;
    if (incx > 0) {
        kx = 0;
    } else {
        kx = -(lenx - 1) * incx;
    }
    if (incy > 0) {
        ky = 0;
    } else {
        ky = -(leny - 1) * incy;
    }

    // First form y := beta*y
    if (beta != 1.0) {
        if (incy == 1) {
            if (beta == 0.0) {
                for (int i = 0; i < leny; i++) {
                    y[i] = 0.0;
                }
            } else {
                for (int i = 0; i < leny; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            int iy = ky;
            if (beta == 0.0) {
                for (int i = 0; i < leny; i++) {
                    y[iy] = 0.0;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < leny; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }

    if (alpha == 0.0) return;

    int kup1 = ku + 1;
    if (trans == 0) {  // 'N' - Form y := alpha*A*x + y
        int jx = kx;
        if (incy == 1) {
            for (int j = 0; j < n; j++) {
                double temp = alpha * x[jx];
                int k = kup1 - 1 - j;
                int i_start = std::max(0, j - ku);
                int i_end = std::min(m - 1, j + kl);
                
                for (int i = i_start; i <= i_end; i++) {
                    y[i] += temp * a[(k + i) + j * lda];
                }
                jx += incx;
            }
        } else {
            for (int j = 0; j < n; j++) {
                double temp = alpha * x[jx];
                int iy = ky;
                int k = kup1 - 1 - j;
                int i_start = std::max(0, j - ku);
                int i_end = std::min(m - 1, j + kl);
                
                for (int i = i_start; i <= i_end; i++) {
                    y[iy] += temp * a[(k + i) + j * lda];
                    iy += incy;
                }
                jx += incx;
                if (j >= ku) ky += incy;
            }
        }
    } else {  // 'T' or 'C' - Form y := alpha*A**T*x + y
        int jy = ky;
        if (incx == 1) {
            for (int j = 0; j < n; j++) {
                double temp = 0.0;
                int k = kup1 - 1 - j;
                int i_start = std::max(0, j - ku);
                int i_end = std::min(m - 1, j + kl);
                
                for (int i = i_start; i <= i_end; i++) {
                    temp += a[(k + i) + j * lda] * x[i];
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        } else {
            for (int j = 0; j < n; j++) {
                double temp = 0.0;
                int ix = kx;
                int k = kup1 - 1 - j;
                int i_start = std::max(0, j - ku);
                int i_end = std::min(m - 1, j + kl);
                
                for (int i = i_start; i <= i_end; i++) {
                    temp += a[(k + i) + j * lda] * x[ix];
                    ix += incx;
                }
                y[jy] += alpha * temp;
                jy += incy;
                if (j >= ku) kx += incx;
            }
        }
    }
}

} // extern "C"
