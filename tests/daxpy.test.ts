/**
 * Tests for DAXPY function
 */

import { daxpy, initWasm } from '../src/index';

describe('DAXPY', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1, incy=1', () => {
    const n = 4;
    const alpha = 2.0;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([5, 6, 7, 8]);

    const result = daxpy(n, alpha, x, 1, y, 1);

    expect(result[0]).toBeCloseTo(7); // 5 + 2*1
    expect(result[1]).toBeCloseTo(10); // 6 + 2*2
    expect(result[2]).toBeCloseTo(13); // 7 + 2*3
    expect(result[3]).toBeCloseTo(16); // 8 + 2*4
  });

  test('with alpha = 0', () => {
    const n = 3;
    const alpha = 0.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    const result = daxpy(n, alpha, x, 1, y, 1);

    expect(result[0]).toBeCloseTo(4);
    expect(result[1]).toBeCloseTo(5);
    expect(result[2]).toBeCloseTo(6);
  });

  test('with negative alpha', () => {
    const n = 3;
    const alpha = -1.5;
    const x = new Float64Array([2, 4, 6]);
    const y = new Float64Array([10, 20, 30]);

    const result = daxpy(n, alpha, x, 1, y, 1);

    expect(result[0]).toBeCloseTo(7); // 10 + (-1.5)*2
    expect(result[1]).toBeCloseTo(14); // 20 + (-1.5)*4
    expect(result[2]).toBeCloseTo(21); // 30 + (-1.5)*6
  });

  test('with incx = 2', () => {
    const n = 3;
    const alpha = 1.0;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);
    const y = new Float64Array([10, 20, 30]);

    const result = daxpy(n, alpha, x, 2, y, 1);

    expect(result[0]).toBeCloseTo(11); // 10 + 1*1
    expect(result[1]).toBeCloseTo(22); // 20 + 1*2
    expect(result[2]).toBeCloseTo(33); // 30 + 1*3
  });

  test('with incy = 2', () => {
    const n = 3;
    const alpha = 2.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([10, 99, 20, 99, 30, 99]);

    const result = daxpy(n, alpha, x, 1, y, 2);

    expect(result[0]).toBeCloseTo(12); // 10 + 2*1
    expect(result[1]).toBeCloseTo(99); // unchanged
    expect(result[2]).toBeCloseTo(24); // 20 + 2*2
    expect(result[3]).toBeCloseTo(99); // unchanged
    expect(result[4]).toBeCloseTo(36); // 30 + 2*3
    expect(result[5]).toBeCloseTo(99); // unchanged
  });

  test('handles large vectors', () => {
    const n = 1000;
    const alpha = 0.5;
    const x = new Float64Array(n).fill(2.0);
    const y = new Float64Array(n).fill(3.0);

    const result = daxpy(n, alpha, x, 1, y, 1);

    for (let i = 0; i < n; i++) {
      expect(result[i]).toBeCloseTo(4.0); // 3 + 0.5*2
    }
  });

  test('handles unrolled loop (n % 4 != 0)', () => {
    const n = 7;
    const alpha = 1.0;
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([10, 20, 30, 40, 50, 60, 70]);

    const result = daxpy(n, alpha, x, 1, y, 1);

    expect(result[0]).toBeCloseTo(11);
    expect(result[1]).toBeCloseTo(22);
    expect(result[2]).toBeCloseTo(33);
    expect(result[3]).toBeCloseTo(44);
    expect(result[4]).toBeCloseTo(55);
    expect(result[5]).toBeCloseTo(66);
    expect(result[6]).toBeCloseTo(77);
  });

  test('throws error for invalid n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    // n = 0 should be handled gracefully (no throw)
    expect(() => daxpy(0, 1.0, x, 1, y, 1)).not.toThrow();
    // Only negative n should throw
    expect(() => daxpy(-1, 1.0, x, 1, y, 1)).toThrow('n must be positive');
  });

  test('throws error for array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => daxpy(4, 1.0, x, 1, y, 1)).toThrow('x array too small');
  });

  test('works with regular arrays', () => {
    const n = 3;
    const alpha = 2.0;
    const x = [1, 2, 3];
    const y = [4, 5, 6];

    const result = daxpy(n, alpha, x, 1, y, 1);

    expect(result[0]).toBeCloseTo(6);
    expect(result[1]).toBeCloseTo(9);
    expect(result[2]).toBeCloseTo(12);
  });
});

describe('DAXPY - Vector Scale and Add', () => {
  beforeAll(async () => {
    await initWasm();
  });

  it('should compute y = alpha*x + y correctly (basic case)', () => {
    const n = 4;
    const alpha = 2.0;
    const x = [1, 2, 3, 4];
    const y = [10, 20, 30, 40];
    const expected = [12, 24, 36, 48]; // y + alpha*x = [10,20,30,40] + 2*[1,2,3,4]

    daxpy(n, alpha, x, 1, y, 1);

    expect(y).toEqual(expected);
  });

  it('should handle alpha = 0 (early return)', () => {
    const n = 3;
    const alpha = 0.0;
    const x = [1, 2, 3];
    const y = [10, 20, 30];
    const expected = [10, 20, 30]; // y should remain unchanged

    daxpy(n, alpha, x, 1, y, 1);

    expect(y).toEqual(expected);
  });

  it('should handle n <= 0 (early return)', () => {
    const n = 0;
    const alpha = 2.0;
    const x = [1, 2, 3];
    const y = [10, 20, 30];
    const expected = [10, 20, 30]; // y should remain unchanged

    daxpy(n, alpha, x, 1, y, 1);

    expect(y).toEqual(expected);
  });

  it('should handle different increments', () => {
    const n = 2;
    const alpha = 3.0;
    const x = [1, 0, 2, 0]; // effective x = [1, 2] with incx = 2
    const y = [10, 0, 20, 0]; // effective y = [10, 20] with incy = 2
    const expected = [13, 0, 26, 0]; // y[0] = 10 + 3*1 = 13, y[2] = 20 + 3*2 = 26

    daxpy(n, alpha, x, 2, y, 2);

    expect(y).toEqual(expected);
  });

  it('should handle negative increments', () => {
    const n = 2;
    const alpha = 2.0;
    const x = [1, 2]; // Will be accessed in reverse due to negative increment
    const y = [10, 20];

    daxpy(n, alpha, x, -1, y, -1);

    // With negative increments, access pattern is reversed
    // First iteration: y[1] = y[1] + alpha * x[1] = 20 + 2*2 = 24
    // Second iteration: y[0] = y[0] + alpha * x[0] = 10 + 2*1 = 12
    expect(y).toEqual([12, 24]);
  });

  it('should handle unrolled loop optimization (n >= 4)', () => {
    const n = 8;
    const alpha = 0.5;
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const y = [10, 20, 30, 40, 50, 60, 70, 80];
    const expected = [10.5, 21, 31.5, 42, 52.5, 63, 73.5, 84];

    daxpy(n, alpha, x, 1, y, 1);

    expect(y).toEqual(expected);
  });
});
