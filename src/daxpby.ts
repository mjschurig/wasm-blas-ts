/**
 * DAXPBY - Double precision extended AXPY
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Computes y = alpha * x + beta * y (extended AXPY operation)
 *
 * @param n - Number of elements in vectors
 * @param alpha - Scalar multiplier for x
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns The modified y vector
 *
 * @example
 * ```typescript
 * import { daxpby, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3, 4]);
 * const y = new Float64Array([5, 6, 7, 8]);
 * const alpha = 2.0;
 * const beta = 3.0;
 *
 * daxpby(4, alpha, x, 1, beta, y, 1);
 * // y is now [17, 22, 27, 32] (i.e., y = 2*x + 3*y)
 * ```
 */
export function daxpby(
  n: number,
  alpha: number,
  x: Float64Array,
  incx: number = 1,
  beta: number,
  y: Float64Array,
  incy: number = 1
): void {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return;
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);
  const yLen = 1 + (n - 1) * Math.abs(incy);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }

  if (y.length < yLen) {
    throw new Error(`y array too small: expected at least ${yLen}, got ${y.length}`);
  }

  // Allocate memory in WASM
  const xPtr = module._malloc(x.length * 8); // 8 bytes per double
  const yPtr = module._malloc(y.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(y, yPtr / 8);

    // Call the WASM function
    module._daxpby(n, alpha, xPtr, incx, beta, yPtr, incy);

    // Copy result back
    const result = new Float64Array(y.length);
    result.set(module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + y.length));

    // Copy back to original array regardless of type
    y.set(result);
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
  }
}
