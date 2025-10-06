/**
 * DTRSV - Double precision triangular solve
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Solves triangular system: A*x = b or A^T*x = b (b is overwritten by x)
 *
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param trans - 'N': A*x = b, 'T'/'C': A^T*x = b
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param n - Order of the matrix A
 * @param a - Triangular matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @param x - Input vector b, output vector x (Float64Array or number[])
 * @param incx - Storage spacing between elements of x (default: 1)
 * @returns The solution vector x
 *
 * @example
 * ```typescript
 * import { dtrsv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x3 upper triangular matrix A
 * const A = new Float64Array([2, 0, 0, 1, 3, 0, 2, 1, 4]); // column-major
 * const b = new Float64Array([4, 9, 16]); // right-hand side
 *
 * dtrsv('U', 'N', 'N', 3, A, 3, b, 1);
 * // b now contains the solution x where A * x = original_b
 * ```
 */
export function dtrsv(
  uplo: string,
  trans: string,
  diag: string,
  n: number,
  a: Float64Array | number[],
  lda: number,
  x: Float64Array | number[],
  incx: number = 1
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (n < 0) {
    throw new Error('n must be non-negative');
  }
  if (lda < Math.max(1, n)) {
    throw new Error(`lda must be at least max(1, n) = ${Math.max(1, n)}, got ${lda}`);
  }

  const xLen = 1 + (n - 1) * Math.abs(incx);

  if (x.length < xLen) {
    throw new Error(`x array too small: expected at least ${xLen}, got ${x.length}`);
  }
  if (a.length < lda * n) {
    throw new Error(`a array too small: expected at least ${lda * n}, got ${a.length}`);
  }

  // Convert to Float64Array if necessary
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);
  const xArray = x instanceof Float64Array ? x : new Float64Array(x);

  // Allocate memory in WASM
  const aPtr = module._malloc(aArray.length * 8);
  const xPtr = module._malloc(xArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(xArray, xPtr / 8);

    // Call the WASM function
    const uploChar = uplo.charCodeAt(0);
    const transChar = trans.charCodeAt(0);
    const diagChar = diag.charCodeAt(0);
    module._dtrsv(uploChar, transChar, diagChar, n, aPtr, lda, xPtr, incx);

    // Copy result back
    const result = new Float64Array(xArray.length);
    result.set(module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + xArray.length));

    // Copy back to original array regardless of type
    if (x instanceof Float64Array) {
      x.set(result);
    } else {
      for (let i = 0; i < result.length; i++) {
        x[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(xPtr);
  }
}
