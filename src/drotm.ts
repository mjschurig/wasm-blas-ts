/**
 * DROTM - Double precision modified Givens rotation
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Applies a modified Givens transformation to vectors x and y
 *
 * @param n - Number of elements in vectors
 * @param x - Input/output vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param y - Input/output vector y (Float64Array or number[])
 * @param incy - Storage spacing between elements of y (default: 1)
 * @param param - Parameter array [flag, h11, h21, h12, h22]
 * @returns Object containing the modified vectors
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
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1,
  param: Float64Array | number[]
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

  // Convert to Float64Array if necessary
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);
  const yArray = y instanceof Float64Array ? y : new Float64Array(y);
  const paramArray = param instanceof Float64Array ? param : new Float64Array(param);

  // Allocate memory in WASM
  const xPtr = module._malloc(xArray.length * 8); // 8 bytes per double
  const yPtr = module._malloc(yArray.length * 8);
  const paramPtr = module._malloc(5 * 8); // 5 parameters

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);
    module.HEAPF64.set(paramArray.slice(0, 5), paramPtr / 8);

    // Call the WASM function
    module._drotm(n, xPtr, incx, yPtr, incy, paramPtr);

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
    module._free(paramPtr);
  }
}
