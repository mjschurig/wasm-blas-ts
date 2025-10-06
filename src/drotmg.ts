/**
 * DROTMG - Double precision modified Givens rotation generation
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Constructs a modified Givens transformation matrix H
 *
 * @param dd1 - Input/output diagonal element
 * @param dd2 - Input/output diagonal element
 * @param dx1 - Input/output vector element
 * @param dy1 - Input vector element
 * @returns Object containing modified values and parameter array
 *
 * @example
 * ```typescript
 * import { drotmg, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const result = drotmg(1.0, 2.0, 3.0, 4.0);
 * // result contains {dd1, dd2, dx1, param} where param is the transformation matrix
 * ```
 */
export function drotmg(
  dd1: number,
  dd2: number,
  dx1: number,
  dy1: number
): { dd1: number; dd2: number; dx1: number; param: Float64Array } {
  const module = getModule();

  // Allocate memory for the scalars and parameter array in WASM
  const dd1Ptr = module._malloc(8); // 8 bytes per double
  const dd2Ptr = module._malloc(8);
  const dx1Ptr = module._malloc(8);
  const paramPtr = module._malloc(5 * 8); // 5 parameters

  try {
    // Set input values
    module.HEAPF64[dd1Ptr / 8] = dd1;
    module.HEAPF64[dd2Ptr / 8] = dd2;
    module.HEAPF64[dx1Ptr / 8] = dx1;

    // Call the WASM function
    module._drotmg(dd1Ptr, dd2Ptr, dx1Ptr, dy1, paramPtr);

    // Read results
    const resultDd1 = module.HEAPF64[dd1Ptr / 8];
    const resultDd2 = module.HEAPF64[dd2Ptr / 8];
    const resultDx1 = module.HEAPF64[dx1Ptr / 8];

    const param = new Float64Array(5);
    param.set(module.HEAPF64.subarray(paramPtr / 8, paramPtr / 8 + 5));

    return { dd1: resultDd1, dd2: resultDd2, dx1: resultDx1, param };
  } finally {
    // Free WASM memory
    module._free(dd1Ptr);
    module._free(dd2Ptr);
    module._free(dx1Ptr);
    module._free(paramPtr);
  }
}
