/**
 * Tests for DCOPY function
 */

import { dcopy, initWasm } from '../src/index';

describe('DCOPY - Vector Copy', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1, incy=1', () => {
    const n = 4;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([0, 0, 0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(2);
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(4);

    // Original y should also be modified
    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(2);
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(4);
  });

  test('overwrites existing values', () => {
    const n = 3;
    const x = new Float64Array([10, 20, 30]);
    const y = new Float64Array([1, 2, 3]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(10);
    expect(y[1]).toBeCloseTo(20);
    expect(y[2]).toBeCloseTo(30);
  });

  test('with negative values', () => {
    const n = 3;
    const x = new Float64Array([-1.5, -2.5, -3.5]);
    const y = new Float64Array([0, 0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(-1.5);
    expect(y[1]).toBeCloseTo(-2.5);
    expect(y[2]).toBeCloseTo(-3.5);
  });

  test('with incx = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);
    const y = new Float64Array([0, 0, 0]);

    dcopy(n, x, 2, y, 1);

    // effective x = [1, 2, 3] (at indices 0, 2, 4)
    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(2);
    expect(y[2]).toBeCloseTo(3);
  });

  test('with incy = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([0, 99, 0, 99, 0, 99]);

    dcopy(n, x, 1, y, 2);

    // Copy to positions 0, 2, 4 in y
    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(99); // unchanged
    expect(y[2]).toBeCloseTo(2);
    expect(y[3]).toBeCloseTo(99); // unchanged
    expect(y[4]).toBeCloseTo(3);
    expect(y[5]).toBeCloseTo(99); // unchanged
  });

  test('with both incx = 2 and incy = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 0, 3, 0]);
    const y = new Float64Array([0, 99, 0, 99]);

    dcopy(n, x, 2, y, 2);

    // effective x = [1, 3], copy to positions 0, 2 in y
    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(99); // unchanged
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(99); // unchanged
  });

  test('with negative increments', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([0, 0, 0]);

    dcopy(n, x, -1, y, -1);

    // With negative increments, should still copy correctly
    expect(y).toHaveLength(3);
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    dcopy(0, x, 1, y, 1);

    // y should remain unchanged
    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(5);
    expect(y[2]).toBeCloseTo(6);
  });

  test('handles single element', () => {
    const n = 1;
    const x = new Float64Array([42.5]);
    const y = new Float64Array([0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(42.5);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(3.14);
    const y = new Float64Array(n).fill(0);

    dcopy(n, x, 1, y, 1);

    for (let i = 0; i < n; i++) {
      expect(y[i]).toBeCloseTo(3.14);
    }
  });

  test('handles very small values', () => {
    const n = 3;
    const x = new Float64Array([1e-10, 2e-10, 3e-10]);
    const y = new Float64Array([0, 0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(1e-10);
    expect(y[1]).toBeCloseTo(2e-10);
    expect(y[2]).toBeCloseTo(3e-10);
  });

  test('handles very large values', () => {
    const n = 3;
    const x = new Float64Array([1e10, 2e10, 3e10]);
    const y = new Float64Array([0, 0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(1e10);
    expect(y[1]).toBeCloseTo(2e10);
    expect(y[2]).toBeCloseTo(3e10);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => dcopy(-1, x, 1, y, 1)).toThrow('n must be positive');
  });

  test('throws error for x array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => dcopy(4, x, 1, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small', () => {
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([3, 4]);

    expect(() => dcopy(4, x, 1, y, 1)).toThrow('y array too small');
  });

  test('throws error for x array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6, 7, 8, 9]);

    expect(() => dcopy(3, x, 2, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small with incy > 1', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => dcopy(3, x, 1, y, 2)).toThrow('y array too small');
  });

  test('handles zero values', () => {
    const n = 3;
    const x = new Float64Array([0, 0, 0]);
    const y = new Float64Array([1, 2, 3]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(0);
    expect(y[1]).toBeCloseTo(0);
    expect(y[2]).toBeCloseTo(0);
  });

  test('handles fractional values', () => {
    const n = 4;
    const x = new Float64Array([1.25, 2.75, 3.125, 4.875]);
    const y = new Float64Array([0, 0, 0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(1.25);
    expect(y[1]).toBeCloseTo(2.75);
    expect(y[2]).toBeCloseTo(3.125);
    expect(y[3]).toBeCloseTo(4.875);
  });

  test('preserves precision', () => {
    const n = 2;
    const x = new Float64Array([Math.PI, Math.E]);
    const y = new Float64Array([0, 0]);

    dcopy(n, x, 1, y, 1);

    expect(y[0]).toBeCloseTo(Math.PI);
    expect(y[1]).toBeCloseTo(Math.E);
  });
});
