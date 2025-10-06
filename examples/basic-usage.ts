/**
 * Basic usage example of wasm-blas-ts
 */

import { initWasm, daxpy } from '../src/index';

async function main() {
  // Initialize the WebAssembly module
  console.log('Initializing WASM module...');
  await initWasm();
  console.log('WASM module initialized!\n');

  // Example 1: Basic DAXPY operation
  console.log('Example 1: Basic DAXPY (y = alpha * x + y)');
  const x1 = new Float64Array([1, 2, 3, 4]);
  const y1 = new Float64Array([5, 6, 7, 8]);
  const alpha1 = 2.0;

  console.log('Input:');
  console.log(`  x = [${x1.join(', ')}]`);
  console.log(`  y = [${y1.join(', ')}]`);
  console.log(`  alpha = ${alpha1}`);

  daxpy(4, alpha1, x1, 1, y1, 1);

  console.log('Output:');
  console.log(`  y = [${y1.join(', ')}]`);
  console.log('  (Expected: [7, 10, 13, 16])\n');

  // Example 2: Using stride (incx/incy)
  console.log('Example 2: Using stride (incx = 2)');
  const x2 = new Float64Array([1, 99, 2, 99, 3, 99]);
  const y2 = new Float64Array([10, 20, 30]);
  const alpha2 = 1.0;

  console.log('Input:');
  console.log(`  x = [${x2.join(', ')}] (stride 2, actual values: 1, 2, 3)`);
  console.log(`  y = [${y2.join(', ')}]`);
  console.log(`  alpha = ${alpha2}`);

  daxpy(3, alpha2, x2, 2, y2, 1);

  console.log('Output:');
  console.log(`  y = [${y2.join(', ')}]`);
  console.log('  (Expected: [11, 22, 33])\n');

  // Example 3: Large vector
  console.log('Example 3: Large vector operation');
  const n = 10000;
  const x3 = new Float64Array(n).fill(1.0);
  const y3 = new Float64Array(n).fill(2.0);
  const alpha3 = 0.5;

  console.log(`Input: ${n} element vectors`);
  console.log(`  x = [1.0, 1.0, ...] (${n} elements)`);
  console.log(`  y = [2.0, 2.0, ...] (${n} elements)`);
  console.log(`  alpha = ${alpha3}`);

  const startTime = performance.now();
  daxpy(n, alpha3, x3, 1, y3, 1);
  const endTime = performance.now();

  console.log('Output:');
  console.log(`  y[0] = ${y3[0]} (Expected: 2.5)`);
  console.log(`  Computation time: ${(endTime - startTime).toFixed(3)} ms\n`);

  // Example 4: Negative alpha
  console.log('Example 4: Negative alpha (subtraction)');
  const x4 = new Float64Array([1, 2, 3]);
  const y4 = new Float64Array([10, 10, 10]);
  const alpha4 = -1.0;

  console.log('Input:');
  console.log(`  x = [${x4.join(', ')}]`);
  console.log(`  y = [${y4.join(', ')}]`);
  console.log(`  alpha = ${alpha4}`);

  daxpy(3, alpha4, x4, 1, y4, 1);

  console.log('Output:');
  console.log(`  y = [${y4.join(', ')}]`);
  console.log('  (Expected: [9, 8, 7])\n');
}

main().catch(console.error);
