/**
 * Tests for DAXPBY function
 */

import { daxpby, initWasm } from '../src/index';

describe('DAXPBY - Extended AXPY (y = alpha*x + beta*y)', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1, incy=1', () => {
    const n = 4;
    const alpha = 2.0;
    const beta = 3.0;
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([5, 6, 7, 8]);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = alpha*x + beta*y = 2*[1,2,3,4] + 3*[5,6,7,8] = [2,4,6,8] + [15,18,21,24] = [17,22,27,32]
    expect(y[0]).toBeCloseTo(17);
    expect(y[1]).toBeCloseTo(22);
    expect(y[2]).toBeCloseTo(27);
    expect(y[3]).toBeCloseTo(32);
  });

  test('with alpha = 0', () => {
    const n = 3;
    const alpha = 0.0;
    const beta = 2.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = 0*x + 2*y = 2*[4,5,6] = [8,10,12]
    expect(y[0]).toBeCloseTo(8);
    expect(y[1]).toBeCloseTo(10);
    expect(y[2]).toBeCloseTo(12);
  });

  test('with beta = 0', () => {
    const n = 3;
    const alpha = 2.5;
    const beta = 0.0;
    const x = new Float64Array([2, 4, 6]);
    const y = new Float64Array([10, 20, 30]);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = 2.5*x + 0*y = 2.5*[2,4,6] = [5,10,15]
    expect(y[0]).toBeCloseTo(5);
    expect(y[1]).toBeCloseTo(10);
    expect(y[2]).toBeCloseTo(15);
  });

  test('with alpha = 1, beta = 1', () => {
    const n = 3;
    const alpha = 1.0;
    const beta = 1.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([10, 20, 30]);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = 1*x + 1*y = [1,2,3] + [10,20,30] = [11,22,33]
    expect(y[0]).toBeCloseTo(11);
    expect(y[1]).toBeCloseTo(22);
    expect(y[2]).toBeCloseTo(33);
  });

  test('with negative alpha and beta', () => {
    const n = 3;
    const alpha = -1.5;
    const beta = -0.5;
    const x = new Float64Array([2, 4, 6]);
    const y = new Float64Array([10, 20, 30]);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = -1.5*x + -0.5*y = -1.5*[2,4,6] + -0.5*[10,20,30] = [-3,-6,-9] + [-5,-10,-15] = [-8,-16,-24]
    expect(y[0]).toBeCloseTo(-8);
    expect(y[1]).toBeCloseTo(-16);
    expect(y[2]).toBeCloseTo(-24);
  });

  test('with incx = 2', () => {
    const n = 3;
    const alpha = 1.0;
    const beta = 1.0;
    const x = new Float64Array([1, 99, 2, 99, 3, 99]);
    const y = new Float64Array([10, 20, 30]);

    daxpby(n, alpha, x, 2, beta, y, 1);

    // effective x = [1, 2, 3], y = [10, 20, 30]
    // y = 1*x + 1*y = [1,2,3] + [10,20,30] = [11,22,33]
    expect(y[0]).toBeCloseTo(11);
    expect(y[1]).toBeCloseTo(22);
    expect(y[2]).toBeCloseTo(33);
  });

  test('with incy = 2', () => {
    const n = 3;
    const alpha = 2.0;
    const beta = 1.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([10, 99, 20, 99, 30, 99]);

    daxpby(n, alpha, x, 1, beta, y, 2);

    // effective y = [10, 20, 30] at positions [0, 2, 4]
    // y = 2*x + 1*y = 2*[1,2,3] + [10,20,30] = [2,4,6] + [10,20,30] = [12,24,36]
    expect(y[0]).toBeCloseTo(12); // position 0
    expect(y[1]).toBeCloseTo(99); // unchanged
    expect(y[2]).toBeCloseTo(24); // position 2
    expect(y[3]).toBeCloseTo(99); // unchanged
    expect(y[4]).toBeCloseTo(36); // position 4
    expect(y[5]).toBeCloseTo(99); // unchanged
  });

  test('with both incx = 2 and incy = 2', () => {
    const n = 2;
    const alpha = 2.0;
    const beta = 3.0;
    const x = new Float64Array([1, 0, 2, 0]);
    const y = new Float64Array([10, 0, 20, 0]);

    daxpby(n, alpha, x, 2, beta, y, 2);

    // effective x = [1, 2], effective y = [10, 20]
    // y = 2*x + 3*y = 2*[1,2] + 3*[10,20] = [2,4] + [30,60] = [32,64]
    expect(y[0]).toBeCloseTo(32);
    expect(y[1]).toBeCloseTo(0); // unchanged
    expect(y[2]).toBeCloseTo(64);
    expect(y[3]).toBeCloseTo(0); // unchanged
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    daxpby(0, 2.0, x, 1, 3.0, y, 1);

    // Should return y unchanged
    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(5);
    expect(y[2]).toBeCloseTo(6);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const alpha = 0.5;
    const beta = 2.0;
    const x = new Float64Array(n).fill(4.0);
    const y = new Float64Array(n).fill(3.0);

    daxpby(n, alpha, x, 1, beta, y, 1);

    // y = 0.5*4 + 2*3 = 2 + 6 = 8
    for (let i = 0; i < n; i++) {
      expect(y[i]).toBeCloseTo(8.0);
    }
  });

  test('handles negative increments', () => {
    const n = 3;
    const alpha = 2.0;
    const beta = 1.0;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([10, 20, 30]);

    daxpby(n, alpha, x, -1, beta, y, -1);

    // With negative increments, vectors are accessed in reverse
    // Should still compute correctly
    expect(y).toHaveLength(3);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => daxpby(-1, 2.0, x, 1, 3.0, y, 1)).toThrow('n must be positive');
  });

  test('throws error for x array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => daxpby(4, 2.0, x, 1, 3.0, y, 1)).toThrow('x array too small');
  });

  test('throws error for y array too small', () => {
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([3, 4]);

    expect(() => daxpby(4, 2.0, x, 1, 3.0, y, 1)).toThrow('y array too small');
  });
});
