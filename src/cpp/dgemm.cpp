/**
 * DGEMM - Double precision general matrix-matrix multiplication
 * 
 * Computes: C = alpha * op(A) * op(B) + beta * C
 * where op(X) = X or X^T
 * 
 * This is a C++ implementation of the BLAS Level 3 DGEMM routine,
 * based on the reference BLAS implementation from netlib.org
 * 
 * @param transa  'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param transb  'N': op(B) = B, 'T'/'C': op(B) = B^T
 * @param m       Number of rows of op(A) and C
 * @param n       Number of columns of op(B) and C
 * @param k       Number of columns of op(A) and rows of op(B)
 * @param alpha   Scalar multiplier for op(A)*op(B)
 * @param a       Matrix A
 * @param lda     Leading dimension of A
 * @param b       Matrix B  
 * @param ldb     Leading dimension of B
 * @param beta    Scalar multiplier for C
 * @param c       Input/output matrix C
 * @param ldc     Leading dimension of C
 */

extern "C" {

void dgemm(char transa, char transb, int m, int n, int k, double alpha,
           const double* a, int lda, const double* b, int ldb, 
           double beta, double* c, int ldc) {
    
    const double zero = 0.0;
    const double one = 1.0;
    
    // Determine transpose flags
    bool nota = (transa == 'N' || transa == 'n');
    bool notb = (transb == 'N' || transb == 'n');
    
    // Quick return if possible
    if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one)) {
        return;
    }
    
    // Handle beta scaling of C
    if (alpha == zero) {
        if (beta == zero) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = zero;
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    c[i + j * ldc] = beta * c[i + j * ldc];
                }
            }
        }
        return;
    }
    
    // Start the operations
    if (notb) {
        if (nota) {
            // Form C := alpha*A*B + beta*C
            for (int j = 0; j < n; j++) {
                if (beta == zero) {
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                
                for (int l = 0; l < k; l++) {
                    double temp = alpha * b[l + j * ldb];
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] += temp * a[i + l * lda];
                    }
                }
            }
        } else {
            // Form C := alpha*A^T*B + beta*C
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    double temp = zero;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * b[l + j * ldb];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp;
                    } else {
                        c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
                    }
                }
            }
        }
    } else {
        if (nota) {
            // Form C := alpha*A*B^T + beta*C
            for (int j = 0; j < n; j++) {
                if (beta == zero) {
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                
                for (int l = 0; l < k; l++) {
                    double temp = alpha * b[j + l * ldb];
                    for (int i = 0; i < m; i++) {
                        c[i + j * ldc] += temp * a[i + l * lda];
                    }
                }
            }
        } else {
            // Form C := alpha*A^T*B^T + beta*C
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    double temp = zero;
                    for (int l = 0; l < k; l++) {
                        temp += a[l + i * lda] * b[j + l * ldb];
                    }
                    if (beta == zero) {
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
