declare module '*/dist/blas.js' {
  interface EmscriptenModuleOptions {
    onRuntimeInitialized?: () => void;
  }

  interface EmscriptenModule {
    _malloc(size: number): number;
    _free(ptr: number): void;
    _daxpy(n: number, alpha: number, xPtr: number, incx: number, yPtr: number, incy: number): void;
    HEAPF64: Float64Array;
    HEAP8: Int8Array;
    HEAPU8: Uint8Array;
    wasmMemory: WebAssembly.Memory;
  }

  function createBlasModule(options?: EmscriptenModuleOptions): Promise<EmscriptenModule>;
  export = createBlasModule;
}
