#include <algorithm>
#include <cmath>

extern "C" {

void dtpmv(int uplo, int trans, int diag, int n, 
           const double* ap, double* x, int incx) {
    // Quick return if possible
    if (n == 0) return;

    bool nounit = (diag == 0);  // diag == 'N'

    // Set up the start point in X if the increment is not unity
    int kx = 0;
    if (incx <= 0) {
        kx = -(n - 1) * incx;
    } else if (incx != 1) {
        kx = 0;
    }

    // Start the operations. In this version the elements of AP are
    // accessed sequentially with one pass through AP.
    
    if (trans == 0) {  // 'N' - Form x := A*x
        if (uplo == 0) {  // Upper triangle
            int kk = 0;
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    if (x[j] != 0.0) {
                        double temp = x[j];
                        int k = kk;
                        for (int i = 0; i < j; i++) {
                            x[i] += temp * ap[k];
                            k++;
                        }
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    if (x[jx] != 0.0) {
                        double temp = x[jx];
                        int ix = kx;
                        for (int k = kk; k < kk + j; k++) {
                            x[ix] += temp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            }
        } else {  // Lower triangle
            int kk = (n * (n + 1)) / 2 - 1;
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    if (x[j] != 0.0) {
                        double temp = x[j];
                        int k = kk;
                        for (int i = n - 1; i > j; i--) {
                            x[i] += temp * ap[k];
                            k--;
                        }
                        if (nounit) x[j] *= ap[kk - n + j + 1];
                    }
                    kk -= (n - j);
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    if (x[jx] != 0.0) {
                        double temp = x[jx];
                        int ix = kx;
                        for (int k = kk; k > kk - (n - j - 1); k--) {
                            x[ix] += temp * ap[k];
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= ap[kk - n + j + 1];
                    }
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    } else {  // 'T' or 'C' - Form x := A**T*x
        if (uplo == 0) {  // Upper triangle
            int kk = (n * (n + 1)) / 2 - 1;
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[j];
                    if (nounit) temp *= ap[kk];
                    int k = kk - 1;
                    for (int i = j - 1; i >= 0; i--) {
                        temp += ap[k] * x[i];
                        k--;
                    }
                    x[j] = temp;
                    kk -= j + 1;
                }
            } else {
                int jx = kx + (n - 1) * incx;
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[jx];
                    int ix = jx;
                    if (nounit) temp *= ap[kk];
                    for (int k = kk - 1; k >= kk - j; k--) {
                        ix -= incx;
                        temp += ap[k] * x[ix];
                    }
                    x[jx] = temp;
                    jx -= incx;
                    kk -= j + 1;
                }
            }
        } else {  // Lower triangle
            int kk = 0;
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    double temp = x[j];
                    if (nounit) temp *= ap[kk];
                    int k = kk + 1;
                    for (int i = j + 1; i < n; i++) {
                        temp += ap[k] * x[i];
                        k++;
                    }
                    x[j] = temp;
                    kk += (n - j);
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    double temp = x[jx];
                    int ix = jx;
                    if (nounit) temp *= ap[kk];
                    for (int k = kk + 1; k < kk + n - j; k++) {
                        ix += incx;
                        temp += ap[k] * x[ix];
                    }
                    x[jx] = temp;
                    jx += incx;
                    kk += (n - j);
                }
            }
        }
    }
}

} // extern "C"
