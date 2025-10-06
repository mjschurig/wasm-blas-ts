/**
 * DASUM - Double precision sum of absolute values
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Computes the sum of absolute values of vector elements: result = sum(|x[i]|)
 *
 * @param n - Number of elements in vector
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @returns The sum of absolute values
 *
 * @example
 * ```typescript
 * import { dasum, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, -2, 3, -4]);
 *
 * const result = dasum(4, x, 1);
 * // result is 10 (|1| + |-2| + |3| + |-4|)
 * ```
 */
export function dasum(n: number, x: Float64Array, incx: number = 1): number {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0 || incx <= 0) {
    return 0.0;
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }

  // Allocate memory in WASM
  const xPtr = module._malloc(x.length * 8); // 8 bytes per double

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(x, xPtr / 8);

    // Call the WASM function
    const result = module._dasum(n, xPtr, incx);

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
  }
}
