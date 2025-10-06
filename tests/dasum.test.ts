/**
 * Tests for DASUM function
 */

import { dasum, initWasm } from '../src/index';

describe('DASUM - Sum of Absolute Values', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1', () => {
    const n = 4;
    const x = new Float64Array([1, -2, 3, -4]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(10); // |1| + |-2| + |3| + |-4| = 10
  });

  test('with all positive values', () => {
    const n = 3;
    const x = new Float64Array([1.5, 2.5, 3.5]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(7.5);
  });

  test('with all negative values', () => {
    const n = 3;
    const x = new Float64Array([-1.5, -2.5, -3.5]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(7.5);
  });

  test('with mixed positive and negative values', () => {
    const n = 5;
    const x = new Float64Array([1, -2, 0, 4, -5]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(12); // |1| + |-2| + |0| + |4| + |-5| = 12
  });

  test('with incx = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 99, -2, 99, 3, 99]);

    const result = dasum(n, x, 2);

    expect(result).toBeCloseTo(6); // |1| + |-2| + |3| = 6
  });

  test('with incx = 3', () => {
    const n = 2;
    const x = new Float64Array([2, 0, 0, -4, 0, 0]);

    const result = dasum(n, x, 3);

    expect(result).toBeCloseTo(6); // |2| + |-4| = 6
  });

  test('with negative incx', () => {
    const n = 3;
    const x = new Float64Array([1, -2, 3]);

    const result = dasum(n, x, -1);

    expect(result).toBeCloseTo(0); // Reference BLAS returns 0 for incx <= 0
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);

    const result = dasum(0, x, 1);

    expect(result).toBe(0.0);
  });

  test('handles empty vector with n = 0', () => {
    const x = new Float64Array([]);

    const result = dasum(0, x, 1);

    expect(result).toBe(0.0);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(-1.5);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(1500); // 1000 * |(-1.5)| = 1500
  });

  test('handles very small values', () => {
    const n = 4;
    const x = new Float64Array([1e-10, -1e-10, 2e-10, -2e-10]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(6e-10);
  });

  test('handles very large values', () => {
    const n = 3;
    const x = new Float64Array([1e10, -2e10, 3e10]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(6e10);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dasum(-1, x, 1)).toThrow('n must be positive');
  });

  test('throws error for array too small', () => {
    const x = new Float64Array([1, 2]);

    expect(() => dasum(4, x, 1)).toThrow('x array too small');
  });

  test('throws error for array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dasum(3, x, 2)).toThrow('x array too small');
  });

  test('handles zero values', () => {
    const n = 5;
    const x = new Float64Array([0, 0, 0, 0, 0]);

    const result = dasum(n, x, 1);

    expect(result).toBe(0.0);
  });

  test('handles single element', () => {
    const n = 1;
    const x = new Float64Array([-5.5]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(5.5);
  });

  test('handles unrolled loop (n >= 6)', () => {
    const n = 8;
    const x = new Float64Array([1, -1, 2, -2, 3, -3, 4, -4]);

    const result = dasum(n, x, 1);

    expect(result).toBeCloseTo(20); // |1| + |1| + |2| + |2| + |3| + |3| + |4| + |4| = 20
  });
});
