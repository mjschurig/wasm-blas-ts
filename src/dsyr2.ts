/**
 * DSYR2 - Double precision symmetric rank-2 update
 * TypeScript wrapper for WebAssembly implementation
 */

import { Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs symmetric rank-2 update: A := alpha * x * y^T + alpha * y * x^T + A
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param alpha - Scalar multiplier
 * @param x - Input vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param a - Input/output symmetric matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @modifies a - The a matrix is modified in-place
 *
 * @example
 * ```typescript
 * import { dsyr2, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([4, 5, 6]);
 * const A = new Float64Array([1, 0, 0, 2, 3, 0, 4, 5, 6]); // 3x3 symmetric matrix
 * const alpha = 1.0;
 *
 * dsyr2('U', 3, alpha, x, 1, y, 1, A, 3);
 * // A = A + alpha * x * y^T + alpha * y * x^T (upper triangular part updated)
 * ```
 */
export function dsyr2(
  uplo: Triangular,
  n: number,
  alpha: number,
  x: Float64Array,
  incx: number = 1,
  y: Float64Array,
  incy: number = 1,
  a: Float64Array,
  lda: number
): void {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be non-negative');
  }
  if (lda < Math.max(1, n)) {
    throw new Error(`lda must be at least max(1, n) = ${Math.max(1, n)}, got ${lda}`);
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);
  const yLen = 1 + (n - 1) * Math.abs(incy);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }
  if (y.length < yLen) {
    throw new Error(`y array too small: expected at least ${yLen}, got ${y.length}`);
  }
  if (a.length < lda * n) {
    throw new Error(`a array too small: expected at least ${lda * n}, got ${a.length}`);
  }

  // Allocate memory in WASM
  const xPtr = module._malloc(x.length * 8);
  const yPtr = module._malloc(y.length * 8);
  const aPtr = module._malloc(a.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(y, yPtr / 8);
    module.HEAPF64.set(a, aPtr / 8);

    // Call the WASM function
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    module._dsyr2(uploChar, n, alpha, xPtr, incx, yPtr, incy, aPtr, lda);

    // Copy result back to a
    const result = module.HEAPF64.subarray(aPtr / 8, aPtr / 8 + a.length);
    a.set(result);
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
    module._free(aPtr);
  }
}
