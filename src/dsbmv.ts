/**
 * DSBMV - Double precision symmetric band matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs symmetric band matrix-vector multiplication: y := alpha * A * x + beta * y
 * where A is a symmetric band matrix
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param k - Number of super-diagonals of A
 * @param alpha - Scalar multiplier for A*x
 * @param a - Symmetric band matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A (>= k + 1)
 * @param x - Input vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @modifies y - The y vector is modified in-place
 *
 * @example
 * ```typescript
 * import { dsbmv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const A = new Float64Array([...]); // band matrix
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([0, 0, 0]);
 *
 * dsbmv('U', 3, 1, 1.0, A, 2, x, 1, 0.0, y, 1);
 * ```
 */

export function dsbmv(
  uplo: Triangular,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array,
  lda: number,
  x: Float64Array,
  incx: number = 1,
  beta: number,
  y: Float64Array,
  incy: number = 1
): void {
  const module = getModule();

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions and band parameter must be non-negative');
  }
  if (lda < k + 1) {
    throw new Error('lda must be at least k + 1');
  }
  if (incx === 0 || incy === 0) {
    throw new Error('Increments cannot be zero');
  }

  // Input arrays are already Float64Array

  // Validate vector sizes
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);
  const minYSize = incy > 0 ? 1 + (n - 1) * incy : 1 + (n - 1) * Math.abs(incy);

  if (x.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${x.length}`);
  }
  if (y.length < minYSize) {
    throw new Error(`y array is too small: expected at least ${minYSize}, got ${y.length}`);
  }

  // Allocate memory
  const aPtr = module._malloc(a.length * 8);
  const xPtr = module._malloc(x.length * 8);
  const yPtr = module._malloc(y.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(y, yPtr / 8);

    // Convert uplo to integer
    const uploInt = uplo === Triangular.Upper ? 0 : 1;

    // Call BLAS function
    module._dsbmv(uploInt, n, k, alpha, aPtr, lda, xPtr, incx, beta, yPtr, incy);

    // Copy result back to y
    const result = module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + y.length);
    y.set(result);
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(xPtr);
    module._free(yPtr);
  }
}
