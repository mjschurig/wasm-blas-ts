/**
 * DROT - Double precision plane rotation
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Applies a plane rotation to vectors x and y:
 * [x] = [c  s] [x]
 * [y]   [-s c] [y]
 *
 * @param n - Number of elements in vectors
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param c - Cosine of the rotation angle
 * @param s - Sine of the rotation angle
 * @modifies x, y - Both vectors are modified in-place
 *
 * @example
 * ```typescript
 * import { drot, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2]);
 * const y = new Float64Array([3, 4]);
 * const c = Math.cos(Math.PI / 4); // 45 degrees
 * const s = Math.sin(Math.PI / 4);
 *
 * drot(2, x, 1, y, 1, c, s);
 * // x and y are now rotated by 45 degrees
 * ```
 */
export function drot(
  n: number,
  x: Float64Array,
  incx: number = 1,
  y: Float64Array,
  incy: number = 1,
  c: number,
  s: number
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
    module._drot(n, xPtr, incx, yPtr, incy, c, s);

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
