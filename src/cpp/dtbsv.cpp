#include <algorithm>
#include <cmath>

extern "C" {

void dtbsv(int uplo, int trans, int diag, int n, int k,
           const double* a, int lda, double* x, int incx) {
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

    // Start the operations. In this version the elements of A are
    // accessed sequentially with one pass through A.
    
    if (trans == 0) {  // 'N' - Form x := inv(A)*x
        if (uplo == 0) {  // Upper triangle
            int kplus1 = k + 1;
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    if (x[j] != 0.0) {
                        int l = kplus1 - 1 - j;
                        if (nounit) x[j] /= a[kplus1 - 1 + j * lda];
                        double temp = x[j];
                        int i_start = std::max(0, j - k);
                        for (int i = j - 1; i >= i_start; i--) {
                            x[i] -= temp * a[(l + i) + j * lda];
                        }
                    }
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    kx -= incx;
                    if (x[jx] != 0.0) {
                        int ix = kx;
                        int l = kplus1 - 1 - j;
                        if (nounit) x[jx] /= a[kplus1 - 1 + j * lda];
                        double temp = x[jx];
                        int i_start = std::max(0, j - k);
                        for (int i = j - 1; i >= i_start; i--) {
                            x[ix] -= temp * a[(l + i) + j * lda];
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            }
        } else {  // Lower triangle
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    if (x[j] != 0.0) {
                        int l = 1 - j;
                        if (nounit) x[j] /= a[0 + j * lda];
                        double temp = x[j];
                        int i_end = std::min(n - 1, j + k);
                        for (int i = j + 1; i <= i_end; i++) {
                            x[i] -= temp * a[(l + i) + j * lda];
                        }
                    }
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    kx += incx;
                    if (x[jx] != 0.0) {
                        int ix = kx;
                        int l = 1 - j;
                        if (nounit) x[jx] /= a[0 + j * lda];
                        double temp = x[jx];
                        int i_end = std::min(n - 1, j + k);
                        for (int i = j + 1; i <= i_end; i++) {
                            x[ix] -= temp * a[(l + i) + j * lda];
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        }
    } else {  // 'T' or 'C' - Form x := inv(A**T)*x
        if (uplo == 0) {  // Upper triangle
            int kplus1 = k + 1;
            if (incx == 1) {
                for (int j = 0; j < n; j++) {
                    double temp = x[j];
                    int l = kplus1 - 1 - j;
                    int i_start = std::max(0, j - k);
                    for (int i = i_start; i < j; i++) {
                        temp -= a[(l + i) + j * lda] * x[i];
                    }
                    if (nounit) temp /= a[kplus1 - 1 + j * lda];
                    x[j] = temp;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < n; j++) {
                    double temp = x[jx];
                    int ix = kx;
                    int l = kplus1 - 1 - j;
                    int i_start = std::max(0, j - k);
                    for (int i = i_start; i < j; i++) {
                        temp -= a[(l + i) + j * lda] * x[ix];
                        ix += incx;
                    }
                    if (nounit) temp /= a[kplus1 - 1 + j * lda];
                    x[jx] = temp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            }
        } else {  // Lower triangle
            if (incx == 1) {
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[j];
                    int l = 1 - j;
                    int i_end = std::min(n - 1, j + k);
                    for (int i = i_end; i > j; i--) {
                        temp -= a[(l + i) + j * lda] * x[i];
                    }
                    if (nounit) temp /= a[0 + j * lda];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                int jx = kx;
                for (int j = n - 1; j >= 0; j--) {
                    double temp = x[jx];
                    int ix = kx;
                    int l = 1 - j;
                    int i_end = std::min(n - 1, j + k);
                    for (int i = i_end; i > j; i--) {
                        temp -= a[(l + i) + j * lda] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) temp /= a[0 + j * lda];
                    x[jx] = temp;
                    jx -= incx;
                    if ((n - 1 - j) >= k) kx -= incx;
                }
            }
        }
    }
}

} // extern "C"
