/**
 * DDOT - Double precision dot product
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Computes the dot product of two vectors: result = x^T * y
 *
 * @param n - Number of elements in vectors
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns The dot product of x and y
 *
 * @example
 * ```typescript
 * import { ddot, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3, 4]);
 * const y = new Float64Array([5, 6, 7, 8]);
 *
 * const result = ddot(4, x, 1, y, 1);
 * // result is 70 (1*5 + 2*6 + 3*7 + 4*8)
 * ```
 */
export function ddot(
  n: number,
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1
): number {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return 0.0;
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
    const result = module._ddot(n, xPtr, incx, yPtr, incy);

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
  }
}
