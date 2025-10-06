/**
 * Tests for DDOT function
 */

import { ddot, initWasm } from '../src/index';

describe('DDOT - Dot Product', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1, incy=1', () => {
    const n = 4;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([5, 6, 7, 8]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    expect(result).toBeCloseTo(70);
  });

  test('with orthogonal vectors', () => {
    const n = 2;
    const x = new Float64Array([1, 0]);
    const y = new Float64Array([0, 1]);

    const result = ddot(n, x, 1, y, 1);

    // dot product of orthogonal vectors is 0
    expect(result).toBeCloseTo(0);
  });

  test('with identical vectors', () => {
    const n = 3;
    const x = new Float64Array([2, 3, 4]);
    const y = new Float64Array([2, 3, 4]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
    expect(result).toBeCloseTo(29);
  });

  test('with negative values', () => {
    const n = 3;
    const x = new Float64Array([1, -2, 3]);
    const y = new Float64Array([-1, 2, -3]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1*(-1) + (-2)*2 + 3*(-3) = -1 + (-4) + (-9) = -14
    expect(result).toBeCloseTo(-14);
  });

  test('with zero vector', () => {
    const n = 4;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([0, 0, 0, 0]);

    const result = ddot(n, x, 1, y, 1);

    expect(result).toBeCloseTo(0);
  });

  test('with incx = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);
    const y = new Float64Array([4, 5, 6]);

    const result = ddot(n, x, 2, y, 1);

    // effective x = [1, 2, 3], y = [4, 5, 6]
    // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expect(result).toBeCloseTo(32);
  });

  test('with incy = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 99, 5, 99, 6, 99]);

    const result = ddot(n, x, 1, y, 2);

    // x = [1, 2, 3], effective y = [4, 5, 6]
    // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expect(result).toBeCloseTo(32);
  });

  test('with both incx = 2 and incy = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 0, 3, 0]);
    const y = new Float64Array([2, 0, 4, 0]);

    const result = ddot(n, x, 2, y, 2);

    // effective x = [1, 3], effective y = [2, 4]
    // dot product = 1*2 + 3*4 = 2 + 12 = 14
    expect(result).toBeCloseTo(14);
  });

  test('with negative increments', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    const result = ddot(n, x, -1, y, -1);

    // With negative increments, vectors are accessed in reverse
    // Should still compute the same dot product
    expect(result).toBeCloseTo(32); // 1*4 + 2*5 + 3*6 = 32
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    const result = ddot(0, x, 1, y, 1);

    expect(result).toBe(0.0);
  });

  test('handles single element', () => {
    const n = 1;
    const x = new Float64Array([3.5]);
    const y = new Float64Array([2.0]);

    const result = ddot(n, x, 1, y, 1);

    expect(result).toBeCloseTo(7.0);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(2.0);
    const y = new Float64Array(n).fill(3.0);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1000 * (2.0 * 3.0) = 6000
    expect(result).toBeCloseTo(6000);
  });

  test('handles very small values', () => {
    const n = 3;
    const x = new Float64Array([1e-10, 2e-10, 3e-10]);
    const y = new Float64Array([4e-10, 5e-10, 6e-10]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1e-10*4e-10 + 2e-10*5e-10 + 3e-10*6e-10 = 4e-20 + 10e-20 + 18e-20 = 32e-20
    expect(result).toBeCloseTo(32e-20);
  });

  test('handles very large values', () => {
    const n = 2;
    const x = new Float64Array([1e10, 2e10]);
    const y = new Float64Array([3e10, 4e10]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1e10*3e10 + 2e10*4e10 = 3e20 + 8e20 = 11e20
    expect(result).toBeCloseTo(11e20);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => ddot(-1, x, 1, y, 1)).toThrow('n must be positive');
  });

  test('throws error for x array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => ddot(4, x, 1, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small', () => {
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([3, 4]);

    expect(() => ddot(4, x, 1, y, 1)).toThrow('y array too small');
  });

  test('throws error for x array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6, 7, 8, 9]);

    expect(() => ddot(3, x, 2, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small with incy > 1', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => ddot(3, x, 1, y, 2)).toThrow('y array too small');
  });

  test('handles fractional values', () => {
    const n = 3;
    const x = new Float64Array([1.5, 2.5, 3.5]);
    const y = new Float64Array([0.5, 1.5, 2.5]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1.5*0.5 + 2.5*1.5 + 3.5*2.5 = 0.75 + 3.75 + 8.75 = 13.25
    expect(result).toBeCloseTo(13.25);
  });

  test('handles unrolled loop (n >= 5)', () => {
    const n = 8;
    const x = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);

    const result = ddot(n, x, 1, y, 1);

    // dot product = 1*1 + 1*2 + 1*3 + 1*4 + 1*5 + 1*6 + 1*7 + 1*8 = 36
    expect(result).toBeCloseTo(36);
  });
});
