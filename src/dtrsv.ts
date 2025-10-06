/**
 * DTRSV - Double precision triangular solve
 * TypeScript wrapper for WebAssembly implementation
 */

import { Diagonal, Transpose, Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Solves triangular system: A*x = b or A^T*x = b (b is overwritten by x)
 *
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param trans - 'N': A*x = b, 'T'/'C': A^T*x = b
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param n - Order of the matrix A
 * @param a - Triangular matrix A in column-major order (Float64Array)
 * @param lda - Leading dimension of A
 * @param x - Input vector b, output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @modifies x - The x vector is modified in-place with the solution
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
    module._dtrsv(uploChar, transChar, diagChar, n, aPtr, lda, xPtr, incx);

    // Copy result back to x
    const result = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    x.set(result);
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(xPtr);
  }
}
