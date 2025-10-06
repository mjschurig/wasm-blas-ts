/**
 * DAXPY - Double precision A*X Plus Y
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Computes y = alpha * x + y
 *
 * @param n - Number of elements in vectors
 * @param alpha - Scalar multiplier for x
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns The modified y vector
 *
 * @example
 * ```typescript
 * import { daxpy, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3, 4]);
 * const y = new Float64Array([5, 6, 7, 8]);
 * const alpha = 2.0;
 *
 * daxpy(4, alpha, x, 1, y, 1);
 * // y is now [7, 10, 13, 16] (i.e., y = 2*x + y)
 * ```
 */
export function daxpy(
  n: number,
  alpha: number,
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    // Early return for n = 0 - no operation needed
    return y instanceof Float64Array ? y : new Float64Array(y);
  }

  if (alpha === 0) {
    // Early return for alpha = 0 - no operation needed
    return y instanceof Float64Array ? y : new Float64Array(y);
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);
  const yLen = 1 + (n - 1) * Math.abs(incy);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }

  if (y.length < yLen) {
    throw new Error(`y array too small: expected at least ${yLen}, got ${y.length}`);
  }

  // Convert to Float64Array if necessary
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);
  const yArray = y instanceof Float64Array ? y : new Float64Array(y);

  // Allocate memory in WASM
  const xPtr = module._malloc(xArray.length * 8); // 8 bytes per double
  const yPtr = module._malloc(yArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);

    // Call the WASM function
    module._daxpy(n, alpha, xPtr, incx, yPtr, incy);

    // Copy result back
    const result = new Float64Array(yArray.length);
    result.set(module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + yArray.length));

    // Copy back to original array regardless of type
    if (y instanceof Float64Array) {
      y.set(result);
    } else {
      // For regular arrays, copy element by element
      for (let i = 0; i < result.length; i++) {
        y[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
  }
}
