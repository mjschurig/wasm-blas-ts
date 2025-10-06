/**
 * wasm-blas-ts - WebAssembly BLAS implementation for TypeScript
 *
 * A high-performance linear algebra library using WebAssembly
 */

export { initWasm, getModule } from './wasm-module';
export { daxpy } from './daxpy';

// Re-export types
export type { BlasModule } from './wasm-module';
