/**
 * DSWAP - Double precision vector swap
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Swaps two vectors: x <-> y
 *
 * @param n - Number of elements in vectors
 * @param x - Input/output vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @returns Object containing the swapped vectors
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
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1
): { x: Float64Array; y: Float64Array } {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return {
      x: x instanceof Float64Array ? x : new Float64Array(x),
      y: y instanceof Float64Array ? y : new Float64Array(y),
    };
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
    module._dswap(n, xPtr, incx, yPtr, incy);

    // Copy results back
    const resultX = new Float64Array(xArray.length);
    const resultY = new Float64Array(yArray.length);
    resultX.set(module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + xArray.length));
    resultY.set(module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + yArray.length));

    // Copy back to original arrays regardless of type
    if (x instanceof Float64Array) {
      x.set(resultX);
    } else {
      for (let i = 0; i < resultX.length; i++) {
        x[i] = resultX[i];
      }
    }

    if (y instanceof Float64Array) {
      y.set(resultY);
    } else {
      for (let i = 0; i < resultY.length; i++) {
        y[i] = resultY[i];
      }
    }

    return { x: resultX, y: resultY };
  } finally {
    // Free WASM memory
    module._free(xPtr);
    module._free(yPtr);
  }
}
