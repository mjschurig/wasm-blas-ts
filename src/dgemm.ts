/**
 * DGEMM - Double precision general matrix-matrix multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

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
 * @param a - Matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @param b - Matrix B in column-major order (Float64Array or number[])
 * @param ldb - Leading dimension of B
 * @param beta - Scalar multiplier for C
 * @param c - Input/output matrix C in column-major order (Float64Array or number[])
 * @param ldc - Leading dimension of C
 * @returns The modified C matrix
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
  transa: string,
  transb: string,
  m: number,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array | number[],
  lda: number,
  b: Float64Array | number[],
  ldb: number,
  beta: number,
  c: Float64Array | number[],
  ldc: number
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (m < 0 || n < 0 || k < 0) {
    throw new Error('m, n, and k must be non-negative');
  }

  const isTransA = transa.toUpperCase() === 'T' || transa.toUpperCase() === 'C';
  const isTransB = transb.toUpperCase() === 'T' || transb.toUpperCase() === 'C';

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

  // Convert to Float64Array if necessary
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);
  const bArray = b instanceof Float64Array ? b : new Float64Array(b);
  const cArray = c instanceof Float64Array ? c : new Float64Array(c);

  // Allocate memory in WASM
  const aPtr = module._malloc(aArray.length * 8);
  const bPtr = module._malloc(bArray.length * 8);
  const cPtr = module._malloc(cArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(bArray, bPtr / 8);
    module.HEAPF64.set(cArray, cPtr / 8);

    // Call the WASM function
    const transaChar = transa.charCodeAt(0);
    const transbChar = transb.charCodeAt(0);
    module._dgemm(transaChar, transbChar, m, n, k, alpha, aPtr, lda, bPtr, ldb, beta, cPtr, ldc);

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
    module._free(bPtr);
    module._free(cPtr);
  }
}
