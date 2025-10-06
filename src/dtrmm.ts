/**
 * DTRMM - Double precision triangular matrix-matrix multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Diagonal, Side, Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs triangular matrix-matrix multiplication:
 * B := alpha*op(A)*B  or  B := alpha*B*op(A)
 * where op(A) = A or A^T and A is triangular
 *
 * @param side - 'L': B := alpha*op(A)*B, 'R': B := alpha*B*op(A)
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param transa - 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param m - Number of rows of matrix B
 * @param n - Number of columns of matrix B
 * @param alpha - Scalar multiplier
 * @param a - Triangular matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param b - Input/output matrix B in column-major order (Float64Array)
 * @param ldb - Leading dimension of B
 * @modifies b - The b matrix is modified in-place
 *
 * @example
 * ```typescript
 * import { dtrmm, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x3 upper triangular matrix A, 3x2 matrix B
 * const A = new Float64Array([2, 0, 0, 1, 3, 0, 2, 1, 4]); // column-major
 * const B = new Float64Array([1, 2, 3, 4, 5, 6]); // 3x2 matrix
 *
 * dtrmm('L', 'U', 'N', 'N', 3, 2, 1.0, A, 3, B, 3);
 * // B = A * B where A is upper triangular
 * ```
 */
export function dtrmm(
  side: Side,
  uplo: Triangular,
  transa: Transpose,
  diag: Diagonal,
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  lda: number,
  b: Float64Array,
  ldb: number
): void {
  const module = getModule();

  // Handle edge cases
  if (m < 0 || n < 0) {
    throw new Error('m and n must be non-negative');
  }

  const isLeft = side === Side.Left;
  const ka = isLeft ? m : n; // Dimension of triangular matrix A

  if (lda < Math.max(1, ka)) {
    throw new Error(`lda must be at least ${Math.max(1, ka)}, got ${lda}`);
  }
  if (ldb < Math.max(1, m)) {
    throw new Error(`ldb must be at least ${Math.max(1, m)}, got ${ldb}`);
  }

  if (a.length < lda * ka) {
    throw new Error(`a array too small: expected at least ${lda * ka}, got ${a.length}`);
  }
  if (b.length < ldb * n) {
    throw new Error(`b array too small: expected at least ${ldb * n}, got ${b.length}`);
  }

  // Allocate memory in WASM
  const aPtr = module._malloc(a.length * 8);
  const bPtr = module._malloc(b.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(b, bPtr / 8);

    // Call the WASM function
    const sideChar = side === Side.Left ? 0 : 1;
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    const transaChar =
      transa === Transpose.NoTranspose ? 0 : transa === Transpose.Transpose ? 1 : 2;
    const diagChar = diag === Diagonal.NonUnit ? 0 : 1;
    module._dtrmm(sideChar, uploChar, transaChar, diagChar, m, n, alpha, aPtr, lda, bPtr, ldb);

    // Copy result back to b
    const result = module.HEAPF64.subarray(bPtr / 8, bPtr / 8 + b.length);
    b.set(result);
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(bPtr);
  }
}
