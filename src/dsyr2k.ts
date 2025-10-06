/**
 * DSYR2K - Double precision symmetric rank-2k update
 * TypeScript wrapper for WebAssembly implementation
 */

import { Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs symmetric rank-2k update:
 * C := alpha*A*B^T + alpha*B*A^T + beta*C  or  C := alpha*A^T*B + alpha*B^T*A + beta*C
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param trans - 'N': C := alpha*A*B^T + alpha*B*A^T + beta*C, 'T'/'C': C := alpha*A^T*B + alpha*B^T*A + beta*C
 * @param n - Order of matrix C
 * @param k - Number of columns of A and B (if trans='N') or rows of A and B (if trans='T')
 * @param alpha - Scalar multiplier for A*B^T + B*A^T or A^T*B + B^T*A
 * @param a - Matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param b - Matrix B in column-major order (Float64Array)
 * @param ldb - Leading dimension of B
 * @param beta - Scalar multiplier for C
 * @param c - Input/output symmetric matrix C in column-major order (Float64Array)
 * @param ldc - Leading dimension of C
 * @modifies c - The c matrix is modified in-place
 *
 * @example
 * ```typescript
 * import { dsyr2k, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x2 matrices A and B, 3x3 symmetric result C
 * const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 3x2 matrix
 * const B = new Float64Array([7, 8, 9, 10, 11, 12]); // 3x2 matrix
 * const C = new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0]); // 3x3 symmetric matrix
 *
 * dsyr2k('U', 'N', 3, 2, 1.0, A, 3, B, 3, 0.0, C, 3);
 * // C = A*B^T + B*A^T (upper triangular part computed)
 * ```
 */
export function dsyr2k(
  uplo: Triangular,
  trans: Transpose,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array,
  lda: number,
  b: Float64Array,
  ldb: number,
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
  const bRows = isNotrans ? n : k;
  const bCols = isNotrans ? k : n;

  if (lda < Math.max(1, aRows)) {
    throw new Error(`lda must be at least ${Math.max(1, aRows)}, got ${lda}`);
  }
  if (ldb < Math.max(1, bRows)) {
    throw new Error(`ldb must be at least ${Math.max(1, bRows)}, got ${ldb}`);
  }
  if (ldc < Math.max(1, n)) {
    throw new Error(`ldc must be at least ${Math.max(1, n)}, got ${ldc}`);
  }

  if (a.length < lda * aCols) {
    throw new Error(`a array too small: expected at least ${lda * aCols}, got ${a.length}`);
  }
  if (b.length < ldb * bCols) {
    throw new Error(`b array too small: expected at least ${ldb * bCols}, got ${b.length}`);
  }
  if (c.length < ldc * n) {
    throw new Error(`c array too small: expected at least ${ldc * n}, got ${c.length}`);
  }

  // Allocate memory in WASM
  const aPtr = module._malloc(a.length * 8);
  const bPtr = module._malloc(b.length * 8);
  const cPtr = module._malloc(c.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(b, bPtr / 8);
    module.HEAPF64.set(c, cPtr / 8);

    // Call the WASM function
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    const transChar = trans === Transpose.NoTranspose ? 0 : 1;
    module._dsyr2k(uploChar, transChar, n, k, alpha, aPtr, lda, bPtr, ldb, beta, cPtr, ldc);

    // Copy result back to c
    const result = module.HEAPF64.subarray(cPtr / 8, cPtr / 8 + c.length);
    c.set(result);
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(bPtr);
    module._free(cPtr);
  }
}
