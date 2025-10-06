/**
 * DSCAL - Double precision vector scaling
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Scales a vector by a constant: x = alpha * x
 *
 * @param n - Number of elements in vector
 * @param alpha - Scalar multiplier
 * @param x - Input/output vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @returns The modified x vector
 *
 * @example
 * ```typescript
 * import { dscal, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3, 4]);
 * const alpha = 2.5;
 *
 * dscal(4, alpha, x, 1);
 * // x is now [2.5, 5.0, 7.5, 10.0]
 * ```
 */
export function dscal(
  n: number,
  alpha: number,
  x: Float64Array | number[],
  incx: number = 1
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return x instanceof Float64Array ? x : new Float64Array(x);
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }

  // Convert to Float64Array if necessary
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);

  // Allocate memory in WASM
  const xPtr = module._malloc(xArray.length * 8); // 8 bytes per double

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(xArray, xPtr / 8);

    // Call the WASM function
    module._dscal(n, alpha, xPtr, incx);

    // Copy result back
    const result = new Float64Array(xArray.length);
    result.set(module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + xArray.length));

    // Copy back to original array regardless of type
    if (x instanceof Float64Array) {
      x.set(result);
    } else {
      // For regular arrays, copy element by element
      for (let i = 0; i < result.length; i++) {
        x[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
  }
}
