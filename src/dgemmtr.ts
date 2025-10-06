import { getModule } from './wasm-module';

export function dgemmtr(
  uplo: 'U' | 'L',
  transa: 'N' | 'T' | 'C',
  transb: 'N' | 'T' | 'C',
  n: number,
  k: number,
  alpha: number,
  a: Float64Array | number[][],
  lda: number,
  b: Float64Array | number[][],
  ldb: number,
  beta: number,
  c: Float64Array | number[][],
  ldc: number
): Float64Array {
  const module = getModule();

  // Validate inputs
  if (n < 0 || k < 0) {
    throw new Error('Matrix dimensions must be non-negative');
  }

  // Determine matrix dimensions based on transpose options
  const nrowa = transa === 'N' ? n : k;
  const nrowb = transb === 'N' ? k : n;

  if (lda < Math.max(1, nrowa)) {
    throw new Error(`lda must be at least max(1, ${nrowa})`);
  }
  if (ldb < Math.max(1, nrowb)) {
    throw new Error(`ldb must be at least max(1, ${nrowb})`);
  }
  if (ldc < Math.max(1, n)) {
    throw new Error(`ldc must be at least max(1, ${n})`);
  }

  // Convert inputs to Float64Array
  const aArray = Array.isArray(a[0])
    ? new Float64Array((a as number[][]).flat())
    : new Float64Array(a as Float64Array);
  const bArray = Array.isArray(b[0])
    ? new Float64Array((b as number[][]).flat())
    : new Float64Array(b as Float64Array);
  const cArray = Array.isArray(c[0])
    ? new Float64Array((c as number[][]).flat())
    : new Float64Array(c as Float64Array);

  // Allocate memory
  const aPtr = module._malloc(aArray.length * 8);
  const bPtr = module._malloc(bArray.length * 8);
  const cPtr = module._malloc(cArray.length * 8);

  try {
    // Copy data to WASM memory
    module.HEAPF64.set(aArray, aPtr / 8);
    module.HEAPF64.set(bArray, bPtr / 8);
    module.HEAPF64.set(cArray, cPtr / 8);

    // Convert parameters to integers
    const uploInt = uplo === 'U' ? 0 : 1;
    const transaInt = transa === 'N' ? 0 : transa === 'T' ? 1 : 2;
    const transbInt = transb === 'N' ? 0 : transb === 'T' ? 1 : 2;

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

    // Copy result back
    const result = new Float64Array(cArray.length);
    result.set(module.HEAPF64.subarray(cPtr / 8, cPtr / 8 + cArray.length));

    // Copy back to original array if it was passed in
    if (c instanceof Float64Array) {
      for (let i = 0; i < Math.min(c.length, result.length); i++) {
        c[i] = result[i];
      }
    } else if (Array.isArray(c)) {
      if (c.length > 0 && Array.isArray(c[0])) {
        // Handle 2D array
        let idx = 0;
        for (let j = 0; j < c.length; j++) {
          const row = c[j];
          for (let i = 0; i < row.length; i++) {
            if (idx < result.length) {
              row[i] = result[idx++];
            }
          }
        }
      } else {
        // Handle 1D array
        for (let i = 0; i < Math.min(c.length, result.length); i++) {
          (c as unknown as number[])[i] = result[i];
        }
      }
    }

    return result;
  } finally {
    // Free allocated memory
    module._free(aPtr);
    module._free(bPtr);
    module._free(cPtr);
  }
}
