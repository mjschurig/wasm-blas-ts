#include <algorithm>
#include <cmath>

extern "C" {

void dgemmtr(int uplo, int transa, int transb, int n, int k, double alpha,
             const double* a, int lda, const double* b, int ldb, 
             double beta, double* c, int ldc) {
    // Quick return if possible
    if (n == 0) return;

    // Set NOTA and NOTB as true if A and B respectively are not
    // transposed and set NROWA and NROWB as the number of rows of A
    // and B respectively.
    bool nota = (transa == 0);  // 'N'
    bool notb = (transb == 0);  // 'N'
    bool upper = (uplo == 0);   // 'U'
    
    // And if alpha == 0
    if (alpha == 0.0) {
        if (beta == 0.0) {
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                for (int i = istart; i <= istop; i++) {
                    c[i + j * ldc] = 0.0;
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                for (int i = istart; i <= istop; i++) {
                    c[i + j * ldc] = beta * c[i + j * ldc];
                }
            }
        }
        return;
    }

    // Start the operations.
    if (notb) {
        if (nota) {
            // Form C := alpha*A*B + beta*C
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                if (beta == 0.0) {
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] = 0.0;
                    }
                } else if (beta != 1.0) {
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                
                for (int l = 0; l < k; l++) {
                    double temp = alpha * b[l + j * ldb];
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] += temp * a[i + l * lda];
                    }
                }
            }
        } else {
            // Form C := alpha*A**T*B + beta*C
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                for (int i = istart; i <= istop; i++) {
                    double temp = 0.0;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * b[l + j * ldb];
                    }
                    if (beta == 0.0) {
                        c[i + j * ldc] = alpha * temp;
                    } else {
                        c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
                    }
                }
            }
        }
    } else {
        if (nota) {
            // Form C := alpha*A*B**T + beta*C
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                if (beta == 0.0) {
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] = 0.0;
                    }
                } else if (beta != 1.0) {
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                
                for (int l = 0; l < k; l++) {
                    double temp = alpha * b[j + l * ldb];
                    for (int i = istart; i <= istop; i++) {
                        c[i + j * ldc] += temp * a[i + l * lda];
                    }
                }
            }
        } else {
            // Form C := alpha*A**T*B**T + beta*C
            for (int j = 0; j < n; j++) {
                int istart = upper ? 0 : j;
                int istop = upper ? j : n - 1;
                
                for (int i = istart; i <= istop; i++) {
                    double temp = 0.0;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * b[j + l * ldb];
                    }
                    if (beta == 0.0) {
                        c[i + j * ldc] = alpha * temp;
                    } else {
                        c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
                    }
                }
            }
        }
    }
}

} // extern "C"
