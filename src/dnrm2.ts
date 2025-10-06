/**
 * DNRM2 - Double precision Euclidean norm
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Computes the Euclidean norm of a vector: result = sqrt(x^T * x)
 *
 * @param n - Number of elements in vector
 * @param x - Input vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @returns The Euclidean norm of x
 *
 * @example
 * ```typescript
 * import { dnrm2, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([3, 4]);
 *
 * const result = dnrm2(2, x, 1);
 * // result is 5.0 (sqrt(3^2 + 4^2))
 * ```
 */
export function dnrm2(n: number, x: Float64Array | number[], incx: number = 1): number {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be positive');
  }
  if (n === 0) {
    return 0.0;
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
    const result = module._dnrm2(n, xPtr, incx);

    return result;
  } finally {
    // Free WASM memory
    module._free(xPtr);
  }
}
