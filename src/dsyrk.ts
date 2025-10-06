/**
 * DSYRK - Double precision symmetric rank-k update
 * TypeScript wrapper for WebAssembly implementation
 */

import { Transpose, Triangular } from './types';
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
 * @param a - Matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param beta - Scalar multiplier for C
 * @param c - Input/output symmetric matrix C in column-major order (Float64Array)
 * @param ldc - Leading dimension of C
 * @modifies c - The c matrix is modified in-place
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
  uplo: Triangular,
  trans: Transpose,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array,
  lda: number,
  beta: number,
  c: Float64Array,
  ldc: number
): void {
  const module = getModule();

  // Handle edge cases
  if (n < 0 || k < 0) {
    throw new Error('n and k must be non-negative');
  }

  const isNotrans = trans === Transpose.NoTranspose;
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

  // Allocate memory in WASM
  const aPtr = module._malloc(a.length * 8);
  const cPtr = module._malloc(c.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(c, cPtr / 8);

    // Call the WASM function
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    const transChar = trans === Transpose.NoTranspose ? 0 : 1;
    module._dsyrk(uploChar, transChar, n, k, alpha, aPtr, lda, beta, cPtr, ldc);

    // Copy result back to c
    const result = module.HEAPF64.subarray(cPtr / 8, cPtr / 8 + c.length);
    c.set(result);
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(cPtr);
  }
}
