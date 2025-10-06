/**
 * WebAssembly module interface and initialization
 */

export interface BlasModule {
  // Level 1 BLAS functions
  _daxpy(n: number, alpha: number, xPtr: number, incx: number, yPtr: number, incy: number): void;
  _dcopy(n: number, xPtr: number, incx: number, yPtr: number, incy: number): void;
  _ddot(n: number, xPtr: number, incx: number, yPtr: number, incy: number): number;
  _dscal(n: number, alpha: number, xPtr: number, incx: number): void;
  _dasum(n: number, xPtr: number, incx: number): number;
  _dnrm2(n: number, xPtr: number, incx: number): number;
  _dswap(n: number, xPtr: number, incx: number, yPtr: number, incy: number): void;
  _drot(
    n: number,
    xPtr: number,
    incx: number,
    yPtr: number,
    incy: number,
    c: number,
    s: number
  ): void;
  _drotg(aPtr: number, bPtr: number, cPtr: number, sPtr: number): void;
  _drotm(n: number, xPtr: number, incx: number, yPtr: number, incy: number, paramPtr: number): void;
  _daxpby(
    n: number,
    alpha: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _drotmg(dd1Ptr: number, dd2Ptr: number, dx1Ptr: number, dy1: number, paramPtr: number): void;

  // Level 2 BLAS functions
  _dgemv(
    trans: number,
    m: number,
    n: number,
    alpha: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _dger(
    m: number,
    n: number,
    alpha: number,
    xPtr: number,
    incx: number,
    yPtr: number,
    incy: number,
    aPtr: number,
    lda: number
  ): void;
  _dsymv(
    uplo: number,
    n: number,
    alpha: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _dsyr(
    uplo: number,
    n: number,
    alpha: number,
    xPtr: number,
    incx: number,
    aPtr: number,
    lda: number
  ): void;
  _dsyr2(
    uplo: number,
    n: number,
    alpha: number,
    xPtr: number,
    incx: number,
    yPtr: number,
    incy: number,
    aPtr: number,
    lda: number
  ): void;
  _dtrmv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number
  ): void;
  _dtrsv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number
  ): void;
  _dgbmv(
    trans: number,
    m: number,
    n: number,
    kl: number,
    ku: number,
    alpha: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _dsbmv(
    uplo: number,
    n: number,
    k: number,
    alpha: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _dspmv(
    uplo: number,
    n: number,
    alpha: number,
    apPtr: number,
    xPtr: number,
    incx: number,
    beta: number,
    yPtr: number,
    incy: number
  ): void;
  _dspr(uplo: number, n: number, alpha: number, xPtr: number, incx: number, apPtr: number): void;
  _dspr2(
    uplo: number,
    n: number,
    alpha: number,
    xPtr: number,
    incx: number,
    yPtr: number,
    incy: number,
    apPtr: number
  ): void;
  _dtbmv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    k: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number
  ): void;
  _dtbsv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    k: number,
    aPtr: number,
    lda: number,
    xPtr: number,
    incx: number
  ): void;
  _dtpmv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    apPtr: number,
    xPtr: number,
    incx: number
  ): void;
  _dtpsv(
    uplo: number,
    trans: number,
    diag: number,
    n: number,
    apPtr: number,
    xPtr: number,
    incx: number
  ): void;

  // Level 3 BLAS functions
  _dgemm(
    transa: number,
    transb: number,
    m: number,
    n: number,
    k: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number,
    beta: number,
    cPtr: number,
    ldc: number
  ): void;
  _dsymm(
    side: number,
    uplo: number,
    m: number,
    n: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number,
    beta: number,
    cPtr: number,
    ldc: number
  ): void;
  _dsyrk(
    uplo: number,
    trans: number,
    n: number,
    k: number,
    alpha: number,
    aPtr: number,
    lda: number,
    beta: number,
    cPtr: number,
    ldc: number
  ): void;
  _dsyr2k(
    uplo: number,
    trans: number,
    n: number,
    k: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number,
    beta: number,
    cPtr: number,
    ldc: number
  ): void;
  _dtrmm(
    side: number,
    uplo: number,
    transa: number,
    diag: number,
    m: number,
    n: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number
  ): void;
  _dtrsm(
    side: number,
    uplo: number,
    transa: number,
    diag: number,
    m: number,
    n: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number
  ): void;
  _dgemmtr(
    uplo: number,
    transa: number,
    transb: number,
    n: number,
    k: number,
    alpha: number,
    aPtr: number,
    lda: number,
    bPtr: number,
    ldb: number,
    beta: number,
    cPtr: number,
    ldc: number
  ): void;

  // Memory management
  _malloc(size: number): number;
  _free(ptr: number): void;

  // Memory views
  HEAPF64: Float64Array;
  HEAP8: Int8Array;
  HEAPU8: Uint8Array;
  wasmMemory: WebAssembly.Memory;
}

let moduleInstance: BlasModule | null = null;

/**
 * Initialize the WebAssembly module
 */
export async function initWasm(): Promise<BlasModule> {
  if (moduleInstance) {
    return moduleInstance;
  }

  try {
    // Import the Emscripten-generated module
    const { default: createBlasModule } = await import('../dist/blas.js');

    // Create module instance with proper initialization
    const module = await createBlasModule({
      onRuntimeInitialized: function () {
        // This ensures heap arrays are available
        // Emscripten calls updateMemoryViews() internally
      },
    });

    if (!module) {
      throw new Error('Failed to initialize WASM module');
    }

    // The createBlasModule returns a promise that resolves to the module
    // Cast to BlasModule interface
    moduleInstance = module as BlasModule;

    return moduleInstance;
  } catch (error) {
    throw new Error(
      `Failed to load WASM module: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Get the initialized WASM module instance
 */
export function getModule(): BlasModule {
  if (!moduleInstance) {
    throw new Error('WASM module not initialized. Call initWasm() first.');
  }
  return moduleInstance;
}
