/**
 * DSYRK - Double precision symmetric rank-k update
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Performs symmetric rank-k update:
 * C := alpha * A * A^T + beta * C  or  C := alpha * A^T * A + beta * C
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param trans - 'N': C := alpha*A*A^T + beta*C, 'T'/'C': C := alpha*A^T*A + beta*C
 * @param n - Order of matrix C
 * @param k - Number of columns of A (if trans='N') or rows of A (if trans='T')
 * @param alpha - Scalar multiplier for A*A^T or A^T*A
 * @param a - Matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @param beta - Scalar multiplier for C
 * @param c - Input/output symmetric matrix C in column-major order (Float64Array or number[])
 * @param ldc - Leading dimension of C
 * @returns The modified C matrix
 *
 * @example
 * ```typescript
 * import { dsyrk, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x2 matrix A, 3x3 symmetric result C
 * const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 3x2 matrix
 * const C = new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0]); // 3x3 symmetric matrix
 *
 * dsyrk('U', 'N', 3, 2, 1.0, A, 3, 0.0, C, 3);
 * // C = A * A^T (upper triangular part computed)
 * ```
 */
export function dsyrk(
  uplo: string,
  trans: string,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array | number[],
  lda: number,
  beta: number,
  c: Float64Array | number[],
  ldc: number
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (n < 0 || k < 0) {
    throw new Error('n and k must be non-negative');
  }

  const isNotrans = trans.toUpperCase() === 'N';
  const aRows = isNotrans ? n : k;
  const aCols = isNotrans ? k : n;

  if (lda < Math.max(1, aRows)) {
    throw new Error(`lda must be at least ${Math.max(1, aRows)}, got ${lda}`);
  }
  if (ldc < Math.max(1, n)) {
    throw new Error(`ldc must be at least ${Math.max(1, n)}, got ${ldc}`);
  }

  if (a.length < lda * aCols) {
    throw new Error(`a array too small: expected at least ${lda * aCols}, got ${a.length}`);
  }
  if (c.length < ldc * n) {
    throw new Error(`c array too small: expected at least ${ldc * n}, got ${c.length}`);
  }

  // Convert to Float64Array if necessary
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);
  const cArray = c instanceof Float64Array ? c : new Float64Array(c);

  // Allocate memory in WASM
  const aPtr = module._malloc(aArray.length * 8);
  const cPtr = module._malloc(cArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(cArray, cPtr / 8);

    // Call the WASM function
    const uploChar = uplo.charCodeAt(0);
    const transChar = trans.charCodeAt(0);
    module._dsyrk(uploChar, transChar, n, k, alpha, aPtr, lda, beta, cPtr, ldc);

    // Copy result back
    const result = new Float64Array(cArray.length);
    result.set(module.HEAPF64.subarray(cPtr / 8, cPtr / 8 + cArray.length));

    // Copy back to original array regardless of type
    if (c instanceof Float64Array) {
      c.set(result);
    } else {
      for (let i = 0; i < result.length; i++) {
        c[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(cPtr);
  }
}
