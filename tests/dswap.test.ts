/**
 * Tests for DSWAP function
 */

import { dswap, initWasm } from '../src/index';

describe('DSWAP - Vector Swap', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1, incy=1', () => {
    const n = 4;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([5, 6, 7, 8]);

    dswap(n, x, 1, y, 1);

    // x should now contain original y values
    expect(x[0]).toBeCloseTo(5);
    expect(x[1]).toBeCloseTo(6);
    expect(x[2]).toBeCloseTo(7);
    expect(x[3]).toBeCloseTo(8);

    // y should now contain original x values
    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(2);
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(4);

    // Original arrays should also be modified
    expect(x[0]).toBeCloseTo(5);
    expect(x[1]).toBeCloseTo(6);
    expect(x[2]).toBeCloseTo(7);
    expect(x[3]).toBeCloseTo(8);

    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(2);
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(4);
  });

  test('with single element', () => {
    const n = 1;
    const x = new Float64Array([42]);
    const y = new Float64Array([99]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(99);
    expect(y[0]).toBeCloseTo(42);
  });

  test('with negative values', () => {
    const n = 3;
    const x = new Float64Array([-1, -2, -3]);
    const y = new Float64Array([1, 2, 3]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);

    expect(y[0]).toBeCloseTo(-1);
    expect(y[1]).toBeCloseTo(-2);
    expect(y[2]).toBeCloseTo(-3);
  });

  test('with mixed positive and negative values', () => {
    const n = 4;
    const x = new Float64Array([1, -2, 3, -4]);
    const y = new Float64Array([-5, 6, -7, 8]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(-5);
    expect(x[1]).toBeCloseTo(6);
    expect(x[2]).toBeCloseTo(-7);
    expect(x[3]).toBeCloseTo(8);

    expect(y[0]).toBeCloseTo(1);
    expect(y[1]).toBeCloseTo(-2);
    expect(y[2]).toBeCloseTo(3);
    expect(y[3]).toBeCloseTo(-4);
  });

  test('with incx = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);
    const y = new Float64Array([10, 20, 30]);

    dswap(n, x, 2, y, 1);

    // Swap elements at positions 0, 2, 4 in x with positions 0, 1, 2 in y
    expect(x[0]).toBeCloseTo(10); // was 1, now y[0]
    expect(x[1]).toBeCloseTo(99); // unchanged
    expect(x[2]).toBeCloseTo(20); // was 2, now y[1]
    expect(x[3]).toBeCloseTo(99); // unchanged
    expect(x[4]).toBeCloseTo(30); // was 3, now y[2]
    expect(x[5]).toBeCloseTo(99); // unchanged

    expect(y[0]).toBeCloseTo(1); // was 10, now x[0]
    expect(y[1]).toBeCloseTo(2); // was 20, now x[2]
    expect(y[2]).toBeCloseTo(3); // was 30, now x[4]
  });

  test('with incy = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([10, 99, 20, 99, 30, 99]);

    dswap(n, x, 1, y, 2);

    expect(x[0]).toBeCloseTo(10); // was 1, now y[0]
    expect(x[1]).toBeCloseTo(20); // was 2, now y[2]
    expect(x[2]).toBeCloseTo(30); // was 3, now y[4]

    // Swap elements at positions 0, 2, 4 in y with positions 0, 1, 2 in x
    expect(y[0]).toBeCloseTo(1); // was 10, now x[0]
    expect(y[1]).toBeCloseTo(99); // unchanged
    expect(y[2]).toBeCloseTo(2); // was 20, now x[1]
    expect(y[3]).toBeCloseTo(99); // unchanged
    expect(y[4]).toBeCloseTo(3); // was 30, now x[2]
    expect(y[5]).toBeCloseTo(99); // unchanged
  });

  test('with both incx = 2 and incy = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 88, 3, 88]);
    const y = new Float64Array([10, 77, 30, 77]);

    dswap(n, x, 2, y, 2);

    expect(x[0]).toBeCloseTo(10); // was 1, now y[0]
    expect(x[1]).toBeCloseTo(88); // unchanged
    expect(x[2]).toBeCloseTo(30); // was 3, now y[2]
    expect(x[3]).toBeCloseTo(88); // unchanged

    expect(y[0]).toBeCloseTo(1); // was 10, now x[0]
    expect(y[1]).toBeCloseTo(77); // unchanged
    expect(y[2]).toBeCloseTo(3); // was 30, now x[2]
    expect(y[3]).toBeCloseTo(77); // unchanged
  });

  test('with negative increments', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    dswap(n, x, -1, y, -1);

    // With negative increments, should still swap correctly
    expect(x).toHaveLength(3);
    expect(y).toHaveLength(3);
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    dswap(0, x, 1, y, 1);

    // Arrays should remain unchanged
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);

    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(5);
    expect(y[2]).toBeCloseTo(6);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(1.0);
    const y = new Float64Array(n).fill(2.0);

    dswap(n, x, 1, y, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(2.0);
      expect(y[i]).toBeCloseTo(1.0);
    }
  });

  test('handles very small values', () => {
    const n = 2;
    const x = new Float64Array([1e-10, 2e-10]);
    const y = new Float64Array([3e-10, 4e-10]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(3e-10);
    expect(x[1]).toBeCloseTo(4e-10);
    expect(y[0]).toBeCloseTo(1e-10);
    expect(y[1]).toBeCloseTo(2e-10);
  });

  test('handles very large values', () => {
    const n = 2;
    const x = new Float64Array([1e10, 2e10]);
    const y = new Float64Array([3e10, 4e10]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(3e10);
    expect(x[1]).toBeCloseTo(4e10);
    expect(y[0]).toBeCloseTo(1e10);
    expect(y[1]).toBeCloseTo(2e10);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => dswap(-1, x, 1, y, 1)).toThrow('n must be positive');
  });

  test('throws error for x array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => dswap(4, x, 1, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small', () => {
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([3, 4]);

    expect(() => dswap(4, x, 1, y, 1)).toThrow('y array too small');
  });

  test('throws error for x array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6, 7, 8, 9]);

    expect(() => dswap(3, x, 2, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small with incy > 1', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => dswap(3, x, 1, y, 2)).toThrow('y array too small');
  });

  test('handles zero values', () => {
    const n = 3;
    const x = new Float64Array([0, 0, 0]);
    const y = new Float64Array([1, 2, 3]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);

    expect(y[0]).toBeCloseTo(0);
    expect(y[1]).toBeCloseTo(0);
    expect(y[2]).toBeCloseTo(0);
  });

  test('handles fractional values', () => {
    const n = 3;
    const x = new Float64Array([1.25, 2.75, 3.125]);
    const y = new Float64Array([4.875, 5.5, 6.25]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(4.875);
    expect(x[1]).toBeCloseTo(5.5);
    expect(x[2]).toBeCloseTo(6.25);

    expect(y[0]).toBeCloseTo(1.25);
    expect(y[1]).toBeCloseTo(2.75);
    expect(y[2]).toBeCloseTo(3.125);
  });

  test('preserves precision with irrational numbers', () => {
    const n = 2;
    const x = new Float64Array([Math.PI, Math.E]);
    const y = new Float64Array([Math.sqrt(2), Math.sqrt(3)]);

    dswap(n, x, 1, y, 1);

    expect(x[0]).toBeCloseTo(Math.sqrt(2));
    expect(x[1]).toBeCloseTo(Math.sqrt(3));

    expect(y[0]).toBeCloseTo(Math.PI);
    expect(y[1]).toBeCloseTo(Math.E);
  });

  test('handles unrolled loop (larger n)', () => {
    const n = 8;
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]);

    dswap(n, x, 1, y, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo((i + 1) * 10);
      expect(y[i]).toBeCloseTo(i + 1);
    }
  });

  test('double swap returns to original', () => {
    const n = 3;
    const originalX = [1, 2, 3];
    const originalY = [4, 5, 6];
    const x = new Float64Array(originalX);
    const y = new Float64Array(originalY);

    // First swap
    dswap(n, x, 1, y, 1);

    // Second swap should return to original
    dswap(n, x, 1, y, 1);

    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(originalX[i]);
      expect(y[i]).toBeCloseTo(originalY[i]);
    }
  });

  test('swap with identical vectors', () => {
    const n = 3;
    const x = new Float64Array([5, 5, 5]);
    const y = new Float64Array([5, 5, 5]);

    dswap(n, x, 1, y, 1);

    // Should remain unchanged
    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(5);
      expect(y[i]).toBeCloseTo(5);
    }
  });
});
