/**
 * DGBMV - Double precision general band matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Transpose } from './types';
import { getModule } from './wasm-module';

/**
 * Performs band matrix-vector multiplication: y = alpha * op(A) * x + beta * y
 * where op(A) = A or A^T and A is a band matrix
 *
 * @param trans - 'N': y = alpha*A*x + beta*y, 'T'/'C': y = alpha*A^T*x + beta*y
 * @param m - Number of rows of matrix A
 * @param n - Number of columns of matrix A
 * @param kl - Number of sub-diagonals of A
 * @param ku - Number of super-diagonals of A
 * @param alpha - Scalar multiplier for A*x or A^T*x
 * @param a - Band matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A (>= kl + ku + 1)
 * @param x - Input vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @modifies y - The y vector is modified in-place
 *
 * @example
 * ```typescript
 * import { dgbmv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const A = new Float64Array([...]);
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([0, 0]);
 *
 * dgbmv('N', 2, 3, 1, 1, 1.0, A, 3, x, 1, 0.0, y, 1);
 * ```
 */

export function dgbmv(
  trans: Transpose,
  m: number,
  n: number,
  kl: number,
  ku: number,
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
  if (m < 0 || n < 0 || kl < 0 || ku < 0) {
    throw new Error('Matrix dimensions and band parameters must be non-negative');
  }
  if (lda < kl + ku + 1) {
    throw new Error('lda must be at least kl + ku + 1');
  }
  if (incx === 0 || incy === 0) {
    throw new Error('Increments cannot be zero');
  }

  // Input arrays are already Float64Array

  // Determine vector lengths
  const lenx = trans === Transpose.NoTranspose ? n : m;
  const leny = trans === Transpose.NoTranspose ? m : n;

  // Validate vector sizes
  const minXSize = incx > 0 ? 1 + (lenx - 1) * incx : 1 + (lenx - 1) * Math.abs(incx);
  const minYSize = incy > 0 ? 1 + (leny - 1) * incy : 1 + (leny - 1) * Math.abs(incy);

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

    // Convert trans to integer
    const transInt = trans === Transpose.NoTranspose ? 0 : trans === Transpose.Transpose ? 1 : 2;

    // Call BLAS function
    module._dgbmv(transInt, m, n, kl, ku, alpha, aPtr, lda, xPtr, incx, beta, yPtr, incy);

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
