/**
 * wasm-blas-ts - WebAssembly BLAS implementation for TypeScript
 *
 * A high-performance linear algebra library using WebAssembly
 */

export { initWasm, getModule } from './wasm-module';

// Level 1 BLAS functions
export { daxpy } from './daxpy';
export { dcopy } from './dcopy';
export { ddot } from './ddot';
export { dscal } from './dscal';
export { dasum } from './dasum';
export { dnrm2 } from './dnrm2';
export { dswap } from './dswap';
export { drot } from './drot';
export { drotg } from './drotg';
export { drotm } from './drotm';
export { daxpby } from './daxpby';
export { drotmg } from './drotmg';

// Level 2 BLAS functions
export { dgemv } from './dgemv';
export { dger } from './dger';
export { dsymv } from './dsymv';
export { dsyr } from './dsyr';
export { dsyr2 } from './dsyr2';
export { dtrmv } from './dtrmv';
export { dtrsv } from './dtrsv';
export { dgbmv } from './dgbmv';
export { dsbmv } from './dsbmv';
export { dspmv } from './dspmv';
export { dspr } from './dspr';
export { dspr2 } from './dspr2';
export { dtbmv } from './dtbmv';
export { dtbsv } from './dtbsv';
export { dtpmv } from './dtpmv';
export { dtpsv } from './dtpsv';

// Level 3 BLAS functions
export { dgemm } from './dgemm';
export { dsymm } from './dsymm';
export { dsyrk } from './dsyrk';
export { dsyr2k } from './dsyr2k';
export { dtrmm } from './dtrmm';
export { dtrsm } from './dtrsm';
export { dgemmtr } from './dgemmtr';

// Re-export types
export type { BlasModule } from './wasm-module';
