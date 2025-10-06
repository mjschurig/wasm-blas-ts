/**
 * DTRSM - Double precision triangular solve with multiple right-hand sides
 * TypeScript wrapper for WebAssembly implementation
 */

import { getModule } from './wasm-module';

/**
 * Solves triangular system with multiple RHS:
 * op(A)*X = alpha*B  or  X*op(A) = alpha*B
 * where op(A) = A or A^T, A is triangular, and X overwrites B
 *
 * @param side - 'L': op(A)*X = alpha*B, 'R': X*op(A) = alpha*B
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param transa - 'N': op(A) = A, 'T'/'C': op(A) = A^T
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param m - Number of rows of matrix B
 * @param n - Number of columns of matrix B
 * @param alpha - Scalar multiplier
 * @param a - Triangular matrix A in column-major order (Float64Array or number[])
 * @param lda - Leading dimension of A
 * @param b - Input matrix B, output matrix X in column-major order (Float64Array or number[])
 * @param ldb - Leading dimension of B
 * @returns The solution matrix X (overwrites B)
 *
 * @example
 * ```typescript
 * import { dtrsm, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * // 3x3 upper triangular matrix A, 3x2 right-hand side matrix B
 * const A = new Float64Array([2, 0, 0, 1, 3, 0, 2, 1, 4]); // column-major
 * const B = new Float64Array([4, 8, 12, 6, 15, 24]); // 3x2 matrix
 *
 * dtrsm('L', 'U', 'N', 'N', 3, 2, 1.0, A, 3, B, 3);
 * // B now contains the solution X where A * X = original_B
 * ```
 */
export function dtrsm(
  side: string,
  uplo: string,
  transa: string,
  diag: string,
  m: number,
  n: number,
  alpha: number,
  a: Float64Array | number[],
  lda: number,
  b: Float64Array | number[],
  ldb: number
): Float64Array {
  const module = getModule();

  // Handle edge cases
  if (m < 0 || n < 0) {
    throw new Error('m and n must be non-negative');
  }

  const isLeft = side.toUpperCase() === 'L';
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

  // Convert to Float64Array if necessary
  const aArray = a instanceof Float64Array ? a : new Float64Array(a);
  const bArray = b instanceof Float64Array ? b : new Float64Array(b);

  // Allocate memory in WASM
  const aPtr = module._malloc(aArray.length * 8);
  const bPtr = module._malloc(bArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(bArray, bPtr / 8);

    // Call the WASM function
    const sideChar = side.charCodeAt(0);
    const uploChar = uplo.charCodeAt(0);
    const transaChar = transa.charCodeAt(0);
    const diagChar = diag.charCodeAt(0);
    module._dtrsm(sideChar, uploChar, transaChar, diagChar, m, n, alpha, aPtr, lda, bPtr, ldb);

    // Copy result back
    const result = new Float64Array(bArray.length);
    result.set(module.HEAPF64.subarray(bPtr / 8, bPtr / 8 + bArray.length));

    // Copy back to original array regardless of type
    if (b instanceof Float64Array) {
      b.set(result);
    } else {
      for (let i = 0; i < result.length; i++) {
        b[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free WASM memory
    module._free(aPtr);
    module._free(bPtr);
  }
}
