/**
 * DGEMV - Double precision general matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Performs matrix-vector multiplication: y = alpha * A * x + beta * y or y = alpha * A^T * x + beta * y
 *
 * @param trans - 'N': y = alpha*A*x + beta*y, 'T'/'C': y = alpha*A^T*x + beta*y
 * @param m - Number of rows of matrix A
 * @param n - Number of columns of matrix A
 * @param alpha - Scalar multiplier for A*x or A^T*x
 * @param a - Matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A (>= max(1,m))
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns The modified y vector
 *
 * @example
 * ```typescript
 * import { dgemv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 2x3 matrix A in column-major order: [[1,2], [3,4], [5,6]]
 * const A = new Float64Array([1, 3, 2, 4, 5, 6]);
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([0, 0]);
 *
 * dgemv('N', 2, 3, 1.0, A, 2, x, 1, 0.0, y, 1);
 * // y = A * x = [22, 28]
 * ```
 */
export function dgemv(
  trans: string,
  m: number,
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
  if (m < 0 || n < 0) {
    throw new Error('m and n must be non-negative');
  }
  if (lda < Math.max(1, m)) {
    throw new Error(`lda must be at least max(1, m) = ${Math.max(1, m)}, got ${lda}`);
  }

  const isTransposed = trans.toUpperCase() === 'T' || trans.toUpperCase() === 'C';
  const xLen = isTransposed ? m : n;
  const yLen = isTransposed ? n : m;

  if (x.length < 1 + (xLen - 1) * Math.abs(incx)) {
    throw new Error(`x array too small`);
  }
  if (y.length < 1 + (yLen - 1) * Math.abs(incy)) {
    throw new Error(`y array too small`);
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
    const transChar = trans.charCodeAt(0);
    module._dgemv(transChar, m, n, alpha, aPtr, lda, xPtr, incx, beta, yPtr, incy);

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
