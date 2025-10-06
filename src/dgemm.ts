/**
 * DGEMM - Double precision general matrix-matrix multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Transpose } from './types';
import { getModule } from './wasm-module';

/**
 * Performs matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
 * where op(X) = X or X^T
 *
 * @param transa - 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param transb - 'N': op(B) = B, 'T'/'C': op(B) = B^T
 * @param m - Number of rows of op(A) and C
 * @param n - Number of columns of op(B) and C
 * @param k - Number of columns of op(A) and rows of op(B)
 * @param alpha - Scalar multiplier for op(A)*op(B)
 * @param a - Matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param b - Matrix B in column-major order (Float64Array)
 * @param ldb - Leading dimension of B
 * @param beta - Scalar multiplier for C
 * @param c - Input/output matrix C in column-major order (Float64Array)
 * @param ldc - Leading dimension of C
 * @modifies c - The c matrix is modified in-place
 *
 * @example
 * ```typescript
 * import { dgemm, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 2x2 matrices in column-major order
 * const A = new Float64Array([1, 3, 2, 4]); // [[1,2], [3,4]]
 * const B = new Float64Array([5, 7, 6, 8]); // [[5,6], [7,8]]
 * const C = new Float64Array([0, 0, 0, 0]); // [[0,0], [0,0]]
 *
 * dgemm('N', 'N', 2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
 * // C = A * B = [[19,22], [43,50]]
 * ```
 */
export function dgemm(
  transa: Transpose,
  transb: Transpose,
  m: number,
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
  if (m < 0 || n < 0 || k < 0) {
    throw new Error('m, n, and k must be non-negative');
  }

  const isTransA = transa === Transpose.Transpose || transa === Transpose.ConjugateTranspose;
  const isTransB = transb === Transpose.Transpose || transb === Transpose.ConjugateTranspose;

  const aRows = isTransA ? k : m;
  const aCols = isTransA ? m : k;
  const bRows = isTransB ? n : k;
  const bCols = isTransB ? k : n;

  if (lda < Math.max(1, aRows)) {
    throw new Error(`lda must be at least ${Math.max(1, aRows)}, got ${lda}`);
  }
  if (ldb < Math.max(1, bRows)) {
    throw new Error(`ldb must be at least ${Math.max(1, bRows)}, got ${ldb}`);
  }
  if (ldc < Math.max(1, m)) {
    throw new Error(`ldc must be at least ${Math.max(1, m)}, got ${ldc}`);
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
    const transaChar = transa.charCodeAt(0);
    const transbChar = transb.charCodeAt(0);
    module._dgemm(transaChar, transbChar, m, n, k, alpha, aPtr, lda, bPtr, ldb, beta, cPtr, ldc);

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
