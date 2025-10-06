/**
 * WebAssembly module interface and initialization
 */

export interface BlasModule {
  _daxpy(n: number, alpha: number, xPtr: number, incx: number, yPtr: number, incy: number): void;
  _malloc(size: number): number;
  _free(ptr: number): void;
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
