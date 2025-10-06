/**
 * DTBMV - Double precision triangular band matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Diagonal, Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs triangular band matrix-vector multiplication: x := op(A) * x
 * where op(A) = A or A^T and A is triangular and banded
 *
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param trans - 'N': A*x, 'T'/'C': A^T*x
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param n - Order of the matrix A
 * @param k - Number of super-diagonals of A
 * @param a - Triangular band matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A (>= k + 1)
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @modifies x - The x vector is modified in-place
 */

export function dtbmv(
  uplo: Triangular,
  trans: Transpose,
  diag: Diagonal,
  n: number,
  k: number,
  a: Float64Array,
  lda: number,
  x: Float64Array,
  incx: number = 1
): void {
  const module = getModule();

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions and band parameter must be non-negative');
  }
  if (lda < k + 1) {
    throw new Error('lda must be at least k + 1');
  }
  if (incx === 0) {
    throw new Error('Increment cannot be zero');
  }

  // Input arrays are already Float64Array

  // Validate vector size
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);

  if (x.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${x.length}`);
  }

  // Allocate memory
  const aPtr = module._malloc(a.length * 8);
  const xPtr = module._malloc(x.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(x, xPtr / 8);

    // Convert parameters to integers
    const uploInt = uplo === Triangular.Upper ? 0 : 1;
    const transInt = trans === Transpose.NoTranspose ? 0 : trans === Transpose.Transpose ? 1 : 2;
    const diagInt = diag === Diagonal.NonUnit ? 0 : 1;

    // Call BLAS function
    module._dtbmv(uploInt, transInt, diagInt, n, k, aPtr, lda, xPtr, incx);

    // Copy result back to x
    const result = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    x.set(result);
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(xPtr);
  }
}
