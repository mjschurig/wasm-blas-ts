import { getModule } from './wasm-module';

export function dspr2(
  uplo: 'U' | 'L',
  n: number,
  alpha: number,
  x: Float64Array | number[],
  incx: number = 1,
  y: Float64Array | number[],
  incy: number = 1,
  ap: Float64Array | number[]
): Float64Array {
  const module = getModule();

  // Validate inputs
  if (n < 0) {
    throw new Error('Matrix dimension must be non-negative');
  }
  if (incx === 0 || incy === 0) {
    throw new Error('Increments cannot be zero');
  }

  // Convert inputs to Float64Array
  const xArray = new Float64Array(x);
  const yArray = new Float64Array(y);
  const apArray = new Float64Array(ap);

  // Validate packed matrix size
  const expectedApSize = (n * (n + 1)) / 2;
  if (apArray.length < expectedApSize) {
    throw new Error(
      `ap array is too small: expected at least ${expectedApSize}, got ${apArray.length}`
    );
  }

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
  const xPtr = module._malloc(xArray.length * 8);
  const yPtr = module._malloc(yArray.length * 8);
  const apPtr = module._malloc(apArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(xArray, xPtr / 8);
    module.HEAPF64.set(yArray, yPtr / 8);
    module.HEAPF64.set(apArray, apPtr / 8);

    // Convert uplo to integer
    const uploInt = uplo === 'U' ? 0 : 1;

    // Call BLAS function
    module._dspr2(uploInt, n, alpha, xPtr, incx, yPtr, incy, apPtr);

    // Copy result back
    const result = new Float64Array(apArray.length);
    result.set(module.HEAPF64.subarray(apPtr / 8, apPtr / 8 + apArray.length));

    // Copy back to original array if it was passed in
    if (ap instanceof Float64Array || Array.isArray(ap)) {
      for (let i = 0; i < Math.min(ap.length, result.length); i++) {
        ap[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free allocated memory
    module._free(xPtr);
    module._free(yPtr);
    module._free(apPtr);
  }
}
