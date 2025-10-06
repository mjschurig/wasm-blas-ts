/**
 * DGEMMTR - Double precision general matrix-matrix multiplication (triangular result)
 * TypeScript wrapper for WebAssembly implementation
 */

import { Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs general matrix-matrix multiplication storing only triangular part:
 * C := alpha * op(A) * op(B) + beta * C
 * where op(X) = X or X^T, and only uplo part of C is computed
 *
 * @param uplo - 'U': compute upper triangular part, 'L': compute lower triangular part
 * @param transa - 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param transb - 'N': op(B) = B, 'T'/'C': op(B) = B^T
 * @param n - Number of rows and columns of C
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
 * import { dgemmtr, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const A = new Float64Array([1, 2, 3, 4]);
 * const B = new Float64Array([5, 6, 7, 8]);
 * const C = new Float64Array([0, 0, 0, 0]);
 *
 * dgemmtr('U', 'N', 'N', 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
 * // Only upper triangular part of C is computed
 * ```
 */

export function dgemmtr(
  uplo: Triangular,
  transa: Transpose,
  transb: Transpose,
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

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions must be non-negative');
  }

  // Determine matrix dimensions based on transpose options
  const nrowa = transa === Transpose.NoTranspose ? n : k;
  const nrowb = transb === Transpose.NoTranspose ? k : n;

  if (lda < Math.max(1, nrowa)) {
    throw new Error(`lda must be at least max(1, ${nrowa})`);
  }
  if (ldb < Math.max(1, nrowb)) {
    throw new Error(`ldb must be at least max(1, ${nrowb})`);
  }
  if (ldc < Math.max(1, n)) {
    throw new Error(`ldc must be at least max(1, ${n})`);
  }

  // Input arrays are already Float64Array

  // Allocate memory
  const aPtr = module._malloc(a.length * 8);
  const bPtr = module._malloc(b.length * 8);
  const cPtr = module._malloc(c.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(b, bPtr / 8);
    module.HEAPF64.set(c, cPtr / 8);

    // Convert parameters to integers
    const uploInt = uplo === Triangular.Upper ? 0 : 1;
    const transaInt = transa === Transpose.NoTranspose ? 0 : transa === Transpose.Transpose ? 1 : 2;
    const transbInt = transb === Transpose.NoTranspose ? 0 : transb === Transpose.Transpose ? 1 : 2;

    // Call BLAS function
    module._dgemmtr(
      uploInt,
      transaInt,
      transbInt,
      n,
      k,
      alpha,
      aPtr,
      lda,
      bPtr,
      ldb,
      beta,
      cPtr,
      ldc
    );

    // Copy result back to c
    const result = module.HEAPF64.subarray(cPtr / 8, cPtr / 8 + c.length);
    c.set(result);
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(bPtr);
    module._free(cPtr);
  }
}
