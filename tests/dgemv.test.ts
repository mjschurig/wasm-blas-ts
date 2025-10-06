/**
 * Tests for DGEMV function
 */

import { dgemv, initWasm, Transpose } from '../src/index';

describe('DGEMV - General Matrix-Vector Multiplication', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation: y = A*x (trans = N)', () => {
    // 2x3 matrix A in column-major order: [[1,2,3], [4,5,6]]
    const m = 2,
      n = 3;
    const A = new Float64Array([1, 4, 2, 5, 3, 6]); // column-major
    const x = new Float64Array([1, 1, 1]);
    const y = new Float64Array([0, 0]);
    const alpha = 1.0,
      beta = 0.0;

    dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

    // A*x = [[1,2,3], [4,5,6]] * [1,1,1] = [6, 15]
    expect(y[0]).toBeCloseTo(6); // 1*1 + 2*1 + 3*1
    expect(y[1]).toBeCloseTo(15); // 4*1 + 5*1 + 6*1
  });

  test('transpose operation: y = A^T*x (trans = T)', () => {
    // 2x3 matrix A: [[1,2,3], [4,5,6]]
    // A^T is 3x2: [[1,4], [2,5], [3,6]]
    const m = 2,
      n = 3;
    const A = new Float64Array([1, 4, 2, 5, 3, 6]);
    const x = new Float64Array([1, 2]); // length m for transpose
    const y = new Float64Array([0, 0, 0]); // length n for y
    const alpha = 1.0,
      beta = 0.0;

    dgemv(Transpose.Transpose, m, n, alpha, A, m, x, 1, beta, y, 1);

    // A^T*x = [[1,4], [2,5], [3,6]] * [1,2] = [9, 12, 15]
    expect(y[0]).toBeCloseTo(9); // 1*1 + 4*2
    expect(y[1]).toBeCloseTo(12); // 2*1 + 5*2
    expect(y[2]).toBeCloseTo(15); // 3*1 + 6*2
  });

  test('with alpha and beta scaling', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([1, 3, 2, 4]); // [[1,2], [3,4]]
    const x = new Float64Array([1, 1]);
    const y = new Float64Array([10, 20]);
    const alpha = 2.0,
      beta = 0.5;

    dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

    // y = alpha*A*x + beta*y = 2*[3,7] + 0.5*[10,20] = [6,14] + [5,10] = [11,24]
    expect(y[0]).toBeCloseTo(11); // 2*(1+2) + 0.5*10
    expect(y[1]).toBeCloseTo(24); // 2*(3+4) + 0.5*20
  });

  test('with alpha = 0', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([1, 2, 3, 4]);
    const x = new Float64Array([5, 6]);
    const y = new Float64Array([10, 20]);
    const alpha = 0.0,
      beta = 2.0;

    dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

    // y = 0*A*x + 2*y = 2*[10,20] = [20,40]
    expect(y[0]).toBeCloseTo(20);
    expect(y[1]).toBeCloseTo(40);
  });

  test('with beta = 0', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([1, 2, 3, 4]); // [[1,3], [2,4]]
    const x = new Float64Array([2, 3]);
    const y = new Float64Array([100, 200]);
    const alpha = 1.0,
      beta = 0.0;

    dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

    // y = A*x = [1*2+3*3, 2*2+4*3] = [11, 16]
    expect(y[0]).toBeCloseTo(11);
    expect(y[1]).toBeCloseTo(16);
  });

  test('identity matrix multiplication', () => {
    const m = 3,
      n = 3;
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // 3x3 identity
    const x = new Float64Array([5, 7, 9]);
    const y = new Float64Array([0, 0, 0]);
    const alpha = 1.0,
      beta = 0.0;

    dgemv(Transpose.NoTranspose, m, n, alpha, I, m, x, 1, beta, y, 1);

    // I*x = x
    expect(y[0]).toBeCloseTo(5);
    expect(y[1]).toBeCloseTo(7);
    expect(y[2]).toBeCloseTo(9);
  });

  test('with different leading dimension (lda > m)', () => {
    // Store a 2x2 matrix in a 3x3 space
    const m = 2,
      n = 2,
      lda = 3;
    const A = new Float64Array([1, 2, 99, 3, 4, 99]); // [[1,3], [2,4]] with padding
    const x = new Float64Array([1, 1]);
    const y = new Float64Array([0, 0]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, lda, x, 1, 0.0, y, 1);

    // Should ignore padding, compute as [[1,3], [2,4]] * [1,1] = [4,6]
    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(6);
  });

  test('with incx = 2', () => {
    const m = 2,
      n = 3;
    const A = new Float64Array([1, 4, 2, 5, 3, 6]);
    const x = new Float64Array([1, 99, 2, 99, 3, 99]); // effective x = [1,2,3]
    const y = new Float64Array([0, 0]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 2, 0.0, y, 1);

    // A*x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    expect(y[0]).toBeCloseTo(14);
    expect(y[1]).toBeCloseTo(32);
  });

  test('with incy = 2', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([1, 3, 2, 4]);
    const x = new Float64Array([1, 1]);
    const y = new Float64Array([10, 88, 20, 88]); // effective y = [10,20]

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 1.0, y, 2);

    // y = A*x + y = [3,7] + [10,20] = [13,27]
    expect(y[0]).toBeCloseTo(13);
    expect(y[1]).toBeCloseTo(88); // unchanged
    expect(y[2]).toBeCloseTo(27);
    expect(y[3]).toBeCloseTo(88); // unchanged
  });

  test('handles rectangular matrix (m > n)', () => {
    const m = 3,
      n = 2;
    const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 3x2 matrix
    const x = new Float64Array([1, 1]);
    const y = new Float64Array([0, 0, 0]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 0.0, y, 1);

    // A*x = [[1,4], [2,5], [3,6]] * [1,1] = [5, 7, 9]
    expect(y[0]).toBeCloseTo(5);
    expect(y[1]).toBeCloseTo(7);
    expect(y[2]).toBeCloseTo(9);
  });

  test('handles rectangular matrix (m < n)', () => {
    const m = 2,
      n = 3;
    const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 2x3 matrix
    const x = new Float64Array([1, 1, 1]);
    const y = new Float64Array([0, 0]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 0.0, y, 1);

    // A*x = [[1,3,5], [2,4,6]] * [1,1,1] = [9, 12]
    expect(y[0]).toBeCloseTo(9);
    expect(y[1]).toBeCloseTo(12);
  });

  test('throws error for invalid dimensions', () => {
    const A = new Float64Array(4);
    const x = new Float64Array(2);
    const y = new Float64Array(2);

    expect(() => dgemv(Transpose.NoTranspose, -1, 2, 1.0, A, 2, x, 1, 0.0, y, 1)).toThrow();
    expect(() => dgemv(Transpose.NoTranspose, 2, -1, 1.0, A, 2, x, 1, 0.0, y, 1)).toThrow();
  });

  test('modifies original arrays', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([1, 3, 2, 4]);
    const x = new Float64Array([1, 1]);
    const y = new Float64Array([1, 1]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 1.0, y, 1);

    // Original y should be modified: y = A*x + y = [3,7] + [1,1] = [4,8]
    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(8);
  });

  test('handles zero matrix', () => {
    const m = 2,
      n = 2;
    const A = new Float64Array([0, 0, 0, 0]);
    const x = new Float64Array([5, 7]);
    const y = new Float64Array([1, 2]);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 2.0, y, 1);

    // y = 0*x + 2*y = 2*[1,2] = [2,4]
    expect(y[0]).toBeCloseTo(2);
    expect(y[1]).toBeCloseTo(4);
  });

  test('handles large matrices', () => {
    const m = 100,
      n = 50;
    const A = new Float64Array(m * n).fill(1.0); // all ones
    const x = new Float64Array(n).fill(1.0);
    const y = new Float64Array(m).fill(0.0);

    dgemv(Transpose.NoTranspose, m, n, 1.0, A, m, x, 1, 0.0, y, 1);

    // Each element should be n (sum of n ones)
    for (let i = 0; i < m; i++) {
      expect(y[i]).toBeCloseTo(n);
    }
  });

  test('conjugate transpose (C) acts same as transpose (T) for real matrices', () => {
    const m = 2,
      n = 3;
    const A = new Float64Array([1, 4, 2, 5, 3, 6]);
    const x = new Float64Array([1, 2]);
    const y1 = new Float64Array([0, 0, 0]);
    const y2 = new Float64Array([0, 0, 0]);

    dgemv(Transpose.Transpose, m, n, 1.0, A, m, x, 1, 0.0, y1, 1);
    dgemv(Transpose.ConjugateTranspose, m, n, 1.0, A, m, x, 1, 0.0, y2, 1);

    // ys should be identical for real matrices
    for (let i = 0; i < n; i++) {
      expect(y1[i]).toBeCloseTo(y2[i]);
    }
  });
});
