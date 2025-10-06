/**
 * DSYMV - Double precision symmetric matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Performs symmetric matrix-vector multiplication: y := alpha * A * x + beta * y
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param alpha - Scalar multiplier for A*x
 * @param a - Symmetric matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A (>= max(1,n))
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns The modified y vector
 *
 * @example
 * ```typescript
 * import { dsymv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x3 symmetric matrix A (upper triangular stored)
 * const A = new Float64Array([1, 0, 0, 2, 3, 0, 4, 5, 6]); // column-major
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([0, 0, 0]);
 *
 * dsymv('U', 3, 1.0, A, 3, x, 1, 0.0, y, 1);
 * // y = A * x where A is symmetric
 * ```
 */
export function dsymv(
  uplo: string,
  n: number,
  alpha: number,
  a: Float64Array | number[],
  lda: number,
  x: Float64Array | number[],
  incx: number = 1,
  beta: number,
  y: Float64Array | number[],
  incy: number = 1
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
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);
  const yArray = y instanceof Float64Array ? y : new Float64Array(y);

  // Allocate memory in WASM
  const aPtr = module._malloc(aArray.length * 8);
  const xPtr = module._malloc(xArray.length * 8);
  const yPtr = module._malloc(yArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);

    // Call the WASM function
    const uploChar = uplo.charCodeAt(0);
    module._dsymv(uploChar, n, alpha, aPtr, lda, xPtr, incx, beta, yPtr, incy);

    // Copy result back
    const result = new Float64Array(yArray.length);
    result.set(module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + yArray.length));

    // Copy back to original array regardless of type
    if (y instanceof Float64Array) {
      y.set(result);
    } else {
      for (let i = 0; i < result.length; i++) {
        y[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(xPtr);
    module._free(yPtr);
  }
}
