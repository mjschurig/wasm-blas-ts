/**
 * DSYR2 - Double precision symmetric rank-2 update
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Performs symmetric rank-2 update: A := alpha * x * y^T + alpha * y * x^T + A
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param alpha - Scalar multiplier
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param a - Input/output symmetric matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @returns The modified A matrix
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
  uplo: string,
  n: number,
  alpha: number,
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1,
  a: Float64Array | number[],
  lda: number
): Float64Array {
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

  // Convert to Float64Array if necessary
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);
  const yArray = y instanceof Float64Array ? y : new Float64Array(y);
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);

  // Allocate memory in WASM
  const xPtr = module._malloc(xArray.length * 8);
  const yPtr = module._malloc(yArray.length * 8);
  const aPtr = module._malloc(aArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);
    module.HEAPF64.set(aArray, aPtr / 8);

    // Call the WASM function
    const uploChar = uplo.charCodeAt(0);
    module._dsyr2(uploChar, n, alpha, xPtr, incx, yPtr, incy, aPtr, lda);

    // Copy result back
    const result = new Float64Array(aArray.length);
    result.set(module.HEAPF64.subarray(aPtr / 8, aPtr / 8 + aArray.length));

    // Copy back to original array regardless of type
    if (a instanceof Float64Array) {
      a.set(result);
    } else {
      for (let i = 0; i < result.length; i++) {
        a[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
    module._free(aPtr);
  }
}
