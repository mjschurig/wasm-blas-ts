/**
 * DSYMM - Double precision symmetric matrix-matrix multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Side, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs symmetric matrix-matrix multiplication:
 * C := alpha * A * B + beta * C  or  C := alpha * B * A + beta * C
 *
 * @param side - 'L': C := alpha*A*B + beta*C, 'R': C := alpha*B*A + beta*C
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param m - Number of rows of matrix C
 * @param n - Number of columns of matrix C
 * @param alpha - Scalar multiplier for A*B or B*A
 * @param a - Symmetric matrix A in column-major order (Float64Array)
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
 * import { dsymm, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // Symmetric 2x2 matrix A, 2x3 matrix B, 2x3 matrix C
 * const A = new Float64Array([1, 2, 2, 3]); // symmetric matrix
 * const B = new Float64Array([1, 3, 2, 4, 1, 2]); // 2x3 matrix
 * const C = new Float64Array([0, 0, 0, 0, 0, 0]); // 2x3 result
 *
 * dsymm('L', 'U', 2, 3, 1.0, A, 2, B, 2, 0.0, C, 2);
 * // C = A * B where A is symmetric
 * ```
 */
export function dsymm(
  side: Side,
  uplo: Triangular,
  m: number,
  n: number,
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
  if (m < 0 || n < 0) {
    throw new Error('m and n must be non-negative');
  }

  const isLeft = side === Side.Left;
  const ka = isLeft ? m : n; // Dimension of symmetric matrix A

  if (lda < Math.max(1, ka)) {
    throw new Error(`lda must be at least ${Math.max(1, ka)}, got ${lda}`);
  }
  if (ldb < Math.max(1, m)) {
    throw new Error(`ldb must be at least ${Math.max(1, m)}, got ${ldb}`);
  }
  if (ldc < Math.max(1, m)) {
    throw new Error(`ldc must be at least ${Math.max(1, m)}, got ${ldc}`);
  }

  if (a.length < lda * ka) {
    throw new Error(`a array too small: expected at least ${lda * ka}, got ${a.length}`);
  }
  if (b.length < ldb * n) {
    throw new Error(`b array too small: expected at least ${ldb * n}, got ${b.length}`);
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
    const sideChar = side === Side.Left ? 0 : 1;
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    module._dsymm(sideChar, uploChar, m, n, alpha, aPtr, lda, bPtr, ldb, beta, cPtr, ldc);

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
