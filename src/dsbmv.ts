import { getModule } from './wasm-module';

export function dsbmv(
  uplo: 'U' | 'L',
  n: number,
  k: number,
  alpha: number,
  a: Float64Array | number[][],
  lda: number,
  x: Float64Array | number[],
  incx: number = 1,
  beta: number,
  y: Float64Array | number[],
  incy: number = 1
): Float64Array {
  const module = getModule();

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions and band parameter must be non-negative');
  }
  if (lda < k + 1) {
    throw new Error('lda must be at least k + 1');
  }
  if (incx === 0 || incy === 0) {
    throw new Error('Increments cannot be zero');
  }

  // Convert inputs to Float64Array
  const aArray = Array.isArray(a[0])
    ? new Float64Array((a as number[][]).flat())
    : new Float64Array(a as Float64Array);
  const xArray = new Float64Array(x);
  const yArray = new Float64Array(y);

  // Validate vector sizes
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);
  const minYSize = incy > 0 ? 1 + (n - 1) * incy : 1 + (n - 1) * Math.abs(incy);

  if (xArray.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${xArray.length}`);
  }
  if (yArray.length < minYSize) {
    throw new Error(`y array is too small: expected at least ${minYSize}, got ${yArray.length}`);
  }

  // Allocate memory
  const aPtr = module._malloc(aArray.length * 8);
  const xPtr = module._malloc(xArray.length * 8);
  const yPtr = module._malloc(yArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);

    // Convert uplo to integer
    const uploInt = uplo === 'U' ? 0 : 1;

    // Call BLAS function
    module._dsbmv(uploInt, n, k, alpha, aPtr, lda, xPtr, incx, beta, yPtr, incy);

    // Copy result back
    const result = new Float64Array(yArray.length);
    result.set(module.HEAPF64.subarray(yPtr / 8, yPtr / 8 + yArray.length));

    // Copy back to original array if it was passed in
    if (y instanceof Float64Array || Array.isArray(y)) {
      for (let i = 0; i < Math.min(y.length, result.length); i++) {
        y[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(xPtr);
    module._free(yPtr);
  }
}
