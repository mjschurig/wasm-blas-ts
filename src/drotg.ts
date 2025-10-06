/**
 * DROTG - Double precision Givens rotation generation
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Constructs a Givens plane rotation that eliminates the second component of a vector
 * [c  s] [a] = [r]
 * [-s c] [b]   [0]
 *
 * @param a - Input scalar a, overwritten with r
 * @param b - Input scalar b, overwritten with z
 * @returns Object containing {r, z, c, s} where c and s are the rotation parameters
 *
 * @example
 * ```typescript
 * import { drotg, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const result = drotg(3.0, 4.0);
 * // result.r is 5.0 (the magnitude)
 * // result.c and result.s are the cosine and sine of the rotation
 * ```
 */
export function drotg(a: number, b: number): { r: number; z: number; c: number; s: number } {
  const module = getModule();

  // Allocate memory for the scalars in WASM
  const aPtr = module._malloc(8); // 8 bytes per double
  const bPtr = module._malloc(8);
  const cPtr = module._malloc(8);
  const sPtr = module._malloc(8);

  try {
    // Set input values
    module.HEAPF64[aPtr / 8] = a;
    module.HEAPF64[bPtr / 8] = b;

    // Call the WASM function
    module._drotg(aPtr, bPtr, cPtr, sPtr);

    // Read results
    const r = module.HEAPF64[aPtr / 8];
    const z = module.HEAPF64[bPtr / 8];
    const c = module.HEAPF64[cPtr / 8];
    const s = module.HEAPF64[sPtr / 8];

    return { r, z, c, s };
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(bPtr);
    module._free(cPtr);
    module._free(sPtr);
  }
}
