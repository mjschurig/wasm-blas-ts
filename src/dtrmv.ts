/**
 * DTRMV - Double precision triangular matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Diagonal, Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs triangular matrix-vector multiplication: x := A*x or x := A^T*x
 *
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param trans - 'N': x := A*x, 'T'/'C': x := A^T*x
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param n - Order of the matrix A
 * @param a - Triangular matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @modifies x - The x vector is modified in-place
 *
 * @example
 * ```typescript
 * import { dtrmv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x3 upper triangular matrix A
 * const A = new Float64Array([1, 0, 0, 2, 3, 0, 4, 5, 6]); // column-major
 * const x = new Float64Array([1, 2, 3]);
 *
 * dtrmv('U', 'N', 'N', 3, A, 3, x, 1);
 * // x = A * x where A is upper triangular
 * ```
 */
export function dtrmv(
  uplo: Triangular,
  trans: Transpose,
  diag: Diagonal,
  n: number,
  a: Float64Array,
  lda: number,
  x: Float64Array,
  incx: number = 1
): void {
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

  // Allocate memory in WASM
  const aPtr = module._malloc(a.length * 8);
  const xPtr = module._malloc(x.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(a, aPtr / 8);
    module.HEAPF64.set(x, xPtr / 8);

    // Call the WASM function
    const uploChar = uplo === Triangular.Upper ? 0 : 1;
    const transChar = trans === Transpose.NoTranspose ? 0 : trans === Transpose.Transpose ? 1 : 2;
    const diagChar = diag === Diagonal.NonUnit ? 0 : 1;
    module._dtrmv(uploChar, transChar, diagChar, n, aPtr, lda, xPtr, incx);

    // Copy result back to x
    const result = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    x.set(result);
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(xPtr);
  }
}
