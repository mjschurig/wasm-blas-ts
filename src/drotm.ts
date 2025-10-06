/**
 * DROTM - Double precision modified Givens rotation
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Applies a modified Givens transformation to vectors x and y
 *
 * @param n - Number of elements in vectors
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param param - Parameter array [flag, h11, h21, h12, h22]
 * @modifies x, y - Both vectors are modified in-place
 *
 * @example
 * ```typescript
 * import { drotm, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([4, 5, 6]);
 * const param = new Float64Array([-1, 0.5, 0.2, -0.1, 0.8]);
 *
 * drotm(3, x, 1, y, 1, param);
 * // x and y are modified according to the transformation matrix
 * ```
 */
export function drotm(
  n: number,
  x: Float64Array,
  incx: number = 1,
  y: Float64Array,
  incy: number = 1,
  param: Float64Array
): void {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return;
  }

  if (param.length < 5) {
    throw new Error('param array must have at least 5 elements');
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
  const paramPtr = module._malloc(5 * 8); // 5 parameters

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(y, yPtr / 8);
    module.HEAPF64.set(param.slice(0, 5), paramPtr / 8);

    // Call the WASM function
    module._drotm(n, xPtr, incx, yPtr, incy, paramPtr);

    // Copy results back to original arrays
    const resultX = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    const resultY = module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + y.length);
    x.set(resultX);
    y.set(resultY);
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
    module._free(paramPtr);
  }
}
