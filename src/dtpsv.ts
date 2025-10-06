/**
 * DTPSV - Double precision triangular packed solve
 * TypeScript wrapper for WebAssembly implementation
 */

import { Triangular, Transpose, Diagonal } from './types';
import { getModule } from './wasm-module';

/**
 * Solves triangular packed system: A*x = b where A is triangular and packed
 *
 * @param uplo - 'U': upper triangular, 'L': lower triangular
 * @param trans - 'N': A*x = b, 'T'/'C': A^T*x = b
 * @param diag - 'U': unit triangular, 'N': non-unit triangular
 * @param n - Order of the matrix A
 * @param ap - Packed triangular matrix A (Float64Array)
 * @param x - Input/output vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @modifies x - The x vector is modified in-place with the solution
 */

export function dtpsv(
  uplo: Triangular,
  trans: Transpose,
  diag: Diagonal,
  n: number,
  ap: Float64Array,
  x: Float64Array,
  incx: number = 1
): void {
  const module = getModule();

  // Validate inputs
  if (n < 0) {
    throw new Error('Matrix dimension must be non-negative');
  }
  if (incx === 0) {
    throw new Error('Increment cannot be zero');
  }

  // Input arrays are already Float64Array

  // Validate packed matrix size
  const expectedApSize = (n * (n + 1)) / 2;
  if (ap.length < expectedApSize) {
    throw new Error(`ap array is too small: expected at least ${expectedApSize}, got ${ap.length}`);
  }

  // Validate vector size
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);

  if (x.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${x.length}`);
  }

  // Allocate memory
  const apPtr = module._malloc(ap.length * 8);
  const xPtr = module._malloc(x.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(ap, apPtr / 8);
    module.HEAPF64.set(x, xPtr / 8);

    // Convert parameters to integers
    const uploInt = uplo === Triangular.Upper ? 0 : 1;
    const transInt = trans === Transpose.NoTranspose ? 0 : trans === Transpose.Transpose ? 1 : 2;
    const diagInt = diag === Diagonal.NonUnit ? 0 : 1;

    // Call BLAS function
    module._dtpsv(uploInt, transInt, diagInt, n, apPtr, xPtr, incx);

    // Copy result back to x
    const result = module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + x.length);
    x.set(result);
  } finally {
    // Free allocated memory
    module._free(apPtr);
    module._free(xPtr);
  }
}
