/**
 * DSPMV - Double precision symmetric packed matrix-vector multiplication
 * TypeScript wrapper for WebAssembly implementation
 */

import { Triangular } from './types';
import { getModule } from './wasm-module';

/**
 * Performs symmetric packed matrix-vector multiplication: y := alpha * A * x + beta * y
 *
 * @param uplo - 'U': use upper triangular part, 'L': use lower triangular part
 * @param n - Order of the matrix A
 * @param alpha - Scalar multiplier for A*x
 * @param ap - Packed symmetric matrix A (Float64Array)
 * @param x - Input vector x (Float64Array)
 * @param incx - Storage spacing between elements of x (default: 1)
 * @param beta - Scalar multiplier for y
 * @param y - Input/output vector y (Float64Array)
 * @param incy - Storage spacing between elements of y (default: 1)
 * @modifies y - The y vector is modified in-place
 *
 * @example
 * ```typescript
 * import { dspmv, initWasm } from 'wasm-blas-ts';
 *
 * await initWasm();
 *
 * const ap = new Float64Array(6); // 3x3 packed matrix = 6 elements
 * const x = new Float64Array([1, 2, 3]);
 * const y = new Float64Array([0, 0, 0]);
 *
 * dspmv('U', 3, 1.0, ap, x, 1, 0.0, y, 1);
 * // y = A * x where A is symmetric packed
 * ```
 */

export function dspmv(
  uplo: Triangular,
  n: number,
  alpha: number,
  ap: Float64Array,
  x: Float64Array,
  incx: number = 1,
  beta: number,
  y: Float64Array,
  incy: number = 1
): void {
  const module = getModule();

  // Validate inputs
  if (n < 0) {
    throw new Error('Matrix dimension must be non-negative');
  }
  if (incx === 0 || incy === 0) {
    throw new Error('Increments cannot be zero');
  }

  // Input arrays are already Float64Array

  // Validate packed matrix size
  const expectedApSize = (n * (n + 1)) / 2;
  if (ap.length < expectedApSize) {
    throw new Error(`ap array is too small: expected at least ${expectedApSize}, got ${ap.length}`);
  }

  // Validate vector sizes
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);
  const minYSize = incy > 0 ? 1 + (n - 1) * incy : 1 + (n - 1) * Math.abs(incy);

  if (x.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${x.length}`);
  }
  if (y.length < minYSize) {
    throw new Error(`y array is too small: expected at least ${minYSize}, got ${y.length}`);
  }

  // Allocate memory
  const apPtr = module._malloc(ap.length * 8);
  const xPtr = module._malloc(x.length * 8);
  const yPtr = module._malloc(y.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(ap, apPtr / 8);
    module.HEAPF64.set(x, xPtr / 8);
    module.HEAPF64.set(y, yPtr / 8);

    // Convert uplo to integer
    const uploInt = uplo === Triangular.Upper ? 0 : 1;

    // Call BLAS function
    module._dspmv(uploInt, n, alpha, apPtr, xPtr, incx, beta, yPtr, incy);

    // Copy result back to y
    const result = module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + y.length);
    y.set(result);
  } finally {
    // Free allocated memory
    module._free(apPtr);
    module._free(xPtr);
    module._free(yPtr);
  }
}
