/**
 * Tests for DSCAL function
 */

import { dscal, initWasm } from '../src/index';

describe('DSCAL - Vector Scaling', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1', () => {
    const n = 4;
    const alpha = 2.5;
    const x = new Float64Array([1, 2, 3, 4]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(2.5);
    expect(x[1]).toBeCloseTo(5.0);
    expect(x[2]).toBeCloseTo(7.5);
    expect(x[3]).toBeCloseTo(10.0);

    // Original x should also be modified
    expect(x[0]).toBeCloseTo(2.5);
    expect(x[1]).toBeCloseTo(5.0);
    expect(x[2]).toBeCloseTo(7.5);
    expect(x[3]).toBeCloseTo(10.0);
  });

  test('with alpha = 0', () => {
    const n = 3;
    const alpha = 0.0;
    const x = new Float64Array([1, 2, 3]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(0);
    expect(x[1]).toBeCloseTo(0);
    expect(x[2]).toBeCloseTo(0);
  });

  test('with alpha = 1', () => {
    const n = 3;
    const alpha = 1.0;
    const x = new Float64Array([1.5, 2.5, 3.5]);

    dscal(n, alpha, x, 1);

    // Should remain unchanged
    expect(x[0]).toBeCloseTo(1.5);
    expect(x[1]).toBeCloseTo(2.5);
    expect(x[2]).toBeCloseTo(3.5);
  });

  test('with alpha = -1', () => {
    const n = 3;
    const alpha = -1.0;
    const x = new Float64Array([1, 2, 3]);

    dscal(n, alpha, x, 1);

    // Should negate all values
    expect(x[0]).toBeCloseTo(-1);
    expect(x[1]).toBeCloseTo(-2);
    expect(x[2]).toBeCloseTo(-3);
  });

  test('with negative alpha', () => {
    const n = 3;
    const alpha = -2.5;
    const x = new Float64Array([2, 4, 6]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(-5.0);
    expect(x[1]).toBeCloseTo(-10.0);
    expect(x[2]).toBeCloseTo(-15.0);
  });

  test('with fractional alpha', () => {
    const n = 4;
    const alpha = 0.5;
    const x = new Float64Array([2, 4, 6, 8]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);
    expect(x[3]).toBeCloseTo(4);
  });

  test('with mixed positive and negative values', () => {
    const n = 4;
    const alpha = 2.0;
    const x = new Float64Array([1, -2, 3, -4]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(2);
    expect(x[1]).toBeCloseTo(-4);
    expect(x[2]).toBeCloseTo(6);
    expect(x[3]).toBeCloseTo(-8);
  });

  test('with incx = 2', () => {
    const n = 3;
    const alpha = 3.0;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);

    dscal(n, alpha, x, 2);

    // Scale elements at positions 0, 2, 4
    expect(x[0]).toBeCloseTo(3); // 1 * 3
    expect(x[1]).toBeCloseTo(99); // unchanged
    expect(x[2]).toBeCloseTo(6); // 2 * 3
    expect(x[3]).toBeCloseTo(99); // unchanged
    expect(x[4]).toBeCloseTo(9); // 3 * 3
    expect(x[5]).toBeCloseTo(99); // unchanged
  });

  test('with incx = 3', () => {
    const n = 2;
    const alpha = 0.5;
    const x = new Float64Array([4, 0, 0, 8, 0, 0]);

    dscal(n, alpha, x, 3);

    // Scale elements at positions 0, 3
    expect(x[0]).toBeCloseTo(2); // 4 * 0.5
    expect(x[1]).toBeCloseTo(0); // unchanged
    expect(x[2]).toBeCloseTo(0); // unchanged
    expect(x[3]).toBeCloseTo(4); // 8 * 0.5
    expect(x[4]).toBeCloseTo(0); // unchanged
    expect(x[5]).toBeCloseTo(0); // unchanged
  });

  test('with negative incx', () => {
    const n = 3;
    const alpha = 2.0;
    const x = new Float64Array([1, 2, 3]);

    dscal(n, alpha, x, -1);

    // Should still scale all elements
    expect(x).toHaveLength(3);
  });

  test('handles n = 0', () => {
    const alpha = 5.0;
    const x = new Float64Array([1, 2, 3]);

    dscal(0, alpha, x, 1);

    // x should remain unchanged
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);
  });

  test('handles single element', () => {
    const n = 1;
    const alpha = 7.5;
    const x = new Float64Array([4]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(30); // 4 * 7.5
  });

  test('handles large vectors', () => {
    const n = 1000;
    const alpha = 0.1;
    const x = new Float64Array(n).fill(10.0);

    dscal(n, alpha, x, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(1.0); // 10.0 * 0.1
    }
  });

  test('handles very small alpha', () => {
    const n = 3;
    const alpha = 1e-10;
    const x = new Float64Array([1e10, 2e10, 3e10]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(1.0);
    expect(x[1]).toBeCloseTo(2.0);
    expect(x[2]).toBeCloseTo(3.0);
  });

  test('handles very large alpha', () => {
    const n = 3;
    const alpha = 1e10;
    const x = new Float64Array([1e-10, 2e-10, 3e-10]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(1.0);
    expect(x[1]).toBeCloseTo(2.0);
    expect(x[2]).toBeCloseTo(3.0);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dscal(-1, 2.0, x, 1)).toThrow('n must be positive');
  });

  test('throws error for array too small', () => {
    const x = new Float64Array([1, 2]);

    expect(() => dscal(4, 2.0, x, 1)).toThrow('x array too small');
  });

  test('throws error for array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dscal(3, 2.0, x, 2)).toThrow('x array too small');
  });

  test('handles zero values', () => {
    const n = 4;
    const alpha = 5.0;
    const x = new Float64Array([0, 0, 0, 0]);

    dscal(n, alpha, x, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(0);
    }
  });

  test('handles precision with irrational numbers', () => {
    const n = 2;
    const alpha = Math.PI;
    const x = new Float64Array([1, 1 / Math.PI]);

    dscal(n, alpha, x, 1);

    expect(x[0]).toBeCloseTo(Math.PI);
    expect(x[1]).toBeCloseTo(1.0); // (1/π) * π = 1
  });

  test('handles unrolled loop (larger n)', () => {
    const n = 8;
    const alpha = 2.0;
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);

    dscal(n, alpha, x, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo((i + 1) * 2);
    }
  });

  test('idempotent scaling (alpha = 1)', () => {
    const n = 5;
    const alpha = 1.0;
    const originalValues = [1.1, 2.2, 3.3, 4.4, 5.5];
    const x = new Float64Array(originalValues);

    dscal(n, alpha, x, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(originalValues[i]);
    }
  });

  test('double scaling equivalence', () => {
    const n = 3;
    const alpha1 = 2.0;
    const alpha2 = 3.0;
    const originalValues = [1, 2, 3];

    // Scale by alpha1, then by alpha2
    const x1 = new Float64Array(originalValues);
    dscal(n, alpha1, x1, 1);
    dscal(n, alpha2, x1, 1);

    // Scale by (alpha1 * alpha2) directly
    const x2 = new Float64Array(originalValues);
    dscal(n, alpha1 * alpha2, x2, 1);

    for (let i = 0; i < n; i++) {
      expect(x1[i]).toBeCloseTo(x2[i]);
    }
  });
});
