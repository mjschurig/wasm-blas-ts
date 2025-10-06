/**
 * DSWAP - Double precision vector swap
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Swaps two vectors: x <-> y
 *
 * @param n - Number of elements in vectors
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @modifies x, y - Both vectors are modified in-place
 *
 * @example
 * ```typescript
 * import { dswap, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3, 4]);
 * const y = new Float64Array([5, 6, 7, 8]);
 *
 * dswap(4, x, 1, y, 1);
 * // x is now [5, 6, 7, 8], y is now [1, 2, 3, 4]
 * ```
 */
export function dswap(
  n: number,
  x: Float64Array,
  incx: number = 1,
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
    module._dswap(n, xPtr, incx, yPtr, incy);

    // Copy results back to original arrays
    const resultX = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    const resultY = module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + y.length);
    x.set(resultX);
    y.set(resultY);
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
  }
}
