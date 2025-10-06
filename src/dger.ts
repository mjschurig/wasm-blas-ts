/**
 * DGER - Double precision general rank-1 update
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Performs rank-1 update: A := alpha * x * y^T + A
 *
 * @param m - Number of rows of matrix A
 * @param n - Number of columns of matrix A
 * @param alpha - Scalar multiplier
 * @param x - Input vector x (Float64Array or number[]) - m elements
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input vector y (Float64Array or number[]) - n elements
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param a - Input/output matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @returns The modified A matrix
 *
 * @example
 * ```typescript
 * import { dger, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2]); // m=2
 * const y = new Float64Array([3, 4, 5]); // n=3
 * const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 2x3 matrix in column-major
 * const alpha = 1.0;
 *
 * dger(2, 3, alpha, x, 1, y, 1, A, 2);
 * // A = A + alpha * x * y^T
 * ```
 */
export function dger(
  m: number,
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
  if (m < 0 || n < 0) {
    throw new Error('m and n must be non-negative');
  }
  if (lda < Math.max(1, m)) {
    throw new Error(`lda must be at least max(1, m) = ${Math.max(1, m)}, got ${lda}`);
  }

  const xLen = 1 + (m - 1) * Math.abs(incx);
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
    module._dger(m, n, alpha, xPtr, incx, yPtr, incy, aPtr, lda);

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
