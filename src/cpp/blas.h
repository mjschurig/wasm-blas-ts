#ifndef BLAS_H
#define BLAS_H

extern "C" {

/**
 * DAXPY - Double precision A*X Plus Y
 * Computes: y = alpha * x + y
 */
void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);

} // extern "C"

#endif // BLAS_H
