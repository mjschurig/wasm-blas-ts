import { Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * DSPR - Double precision symmetric packed rank-1 update
 * TypeScript wrapper for WebAssembly implementation
 */

/**
 * Performs symmetric packed rank-1 update: A := alpha * x * x^T + A
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param alpha - Scalar multiplier
 * @param x - Input vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param ap - Input/output packed symmetric matrix A (Float64Array)
 * @modifies ap - The ap matrix is modified in-place
 *
 * @example
 * ```typescript
 * import { dspr, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const x = new Float64Array([1, 2, 3]);
 * const ap = new Float64Array(6); // 3x3 packed matrix = 6 elements
 * const alpha = 1.0;
 *
 * dspr('U', 3, alpha, x, 1, ap);
 * // ap = ap + alpha * x * x^T
 * ```
 */

export function dspr(
  uplo: Triangular,
  n: number,
  alpha: number,
  x: Float64Array,
  incx: number = 1,
  ap: Float64Array
): void {
  const module = getModule();

  // Validate inputs
  if (n < 0) {
    throw new Error('Matrix dimension must be non-negative');
  }
  if (incx === 0) {
    throw new Error('Increment cannot be zero');
  }

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
  const xPtr = module._malloc(x.length * 8);
  const apPtr = module._malloc(ap.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(ap, apPtr / 8);

    // Convert uplo to integer
    const uploInt = uplo === Triangular.Upper ? 0 : 1;

    // Call BLAS function
    module._dspr(uploInt, n, alpha, xPtr, incx, apPtr);

    // Copy result back to ap
    const result = module.HEAPF64.subarray(apPtr / 8, apPtr / 8 + ap.length);
    ap.set(result);
  } finally {
    // Free allocated memory
    module._free(xPtr);
    module._free(apPtr);
  }
}
