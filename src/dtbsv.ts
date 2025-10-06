import { getModule } from './wasm-module';

export function dtbsv(
  uplo: 'U' | 'L',
  trans: 'N' | 'T' | 'C',
  diag: 'U' | 'N',
  n: number,
  k: number,
  a: Float64Array | number[][],
  lda: number,
  x: Float64Array | number[],
  incx: number = 1
): Float64Array {
  const module = getModule();

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions and band parameter must be non-negative');
  }
  if (lda < k + 1) {
    throw new Error('lda must be at least k + 1');
  }
  if (incx === 0) {
    throw new Error('Increment cannot be zero');
  }

  // Convert inputs to Float64Array
  const aArray = Array.isArray(a[0])
    ? new Float64Array((a as number[][]).flat())
    : new Float64Array(a as Float64Array);
  const xArray = new Float64Array(x);

  // Validate vector size
  const minXSize = incx > 0 ? 1 + (n - 1) * incx : 1 + (n - 1) * Math.abs(incx);

  if (xArray.length < minXSize) {
    throw new Error(`x array is too small: expected at least ${minXSize}, got ${xArray.length}`);
  }

  // Allocate memory
  const aPtr = module._malloc(aArray.length * 8);
  const xPtr = module._malloc(xArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(xArray, xPtr / 8);

    // Convert parameters to integers
    const uploInt = uplo === 'U' ? 0 : 1;
    const transInt = trans === 'N' ? 0 : trans === 'T' ? 1 : 2;
    const diagInt = diag === 'N' ? 0 : 1;

    // Call BLAS function
    module._dtbsv(uploInt, transInt, diagInt, n, k, aPtr, lda, xPtr, incx);

    // Copy result back
    const result = new Float64Array(xArray.length);
    result.set(module.HEAPF64.subarray(xPtr / 8, xPtr / 8 + xArray.length));

    // Copy back to original array if it was passed in
    if (x instanceof Float64Array || Array.isArray(x)) {
      for (let i = 0; i < Math.min(x.length, result.length); i++) {
        x[i] = result[i];
      }
    }

    return result;
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(xPtr);
  }
}
