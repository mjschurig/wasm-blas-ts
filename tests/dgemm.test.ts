/**
 * Tests for DGEMM function
 */

import { dgemm, initWasm, Transpose } from '../src/index';

describe('DGEMM - General Matrix-Matrix Multiplication', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation: C = A*B (no transpose)', () => {
    // 2x2 matrices: A = [[1,2], [3,4]], B = [[5,6], [7,8]]
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 3, 2, 4]); // column-major
    const B = new Float64Array([5, 7, 6, 8]); // column-major
    const C = new Float64Array([0, 0, 0, 0]);
    const alpha = 1.0,
      beta = 0.0;

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, alpha, A, m, B, k, beta, C, m);

    // C = A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22], [43,50]]
    expect(C[0]).toBeCloseTo(19); // C[0,0]
    expect(C[1]).toBeCloseTo(43); // C[1,0]
    expect(C[2]).toBeCloseTo(22); // C[0,1]
    expect(C[3]).toBeCloseTo(50); // C[1,1]
  });

  test('transpose A: C = A^T*B', () => {
    // A = [[1,2], [3,4]] -> A^T = [[1,3], [2,4]]
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 3, 2, 4]); // original A
    const B = new Float64Array([1, 0, 0, 1]); // identity matrix
    const C = new Float64Array([0, 0, 0, 0]);

    dgemm(Transpose.Transpose, Transpose.NoTranspose, m, n, k, 1.0, A, k, B, k, 0.0, C, m);

    // C = A^T*I = A^T = [[1,3], [2,4]]
    expect(C[0]).toBeCloseTo(1); // C[0,0] = 1
    expect(C[1]).toBeCloseTo(2); // C[1,0] = 2
    expect(C[2]).toBeCloseTo(3); // C[0,1] = 3
    expect(C[3]).toBeCloseTo(4); // C[1,1] = 4
  });

  test('transpose B: C = A*B^T', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 0, 0, 1]); // identity matrix
    const B = new Float64Array([1, 3, 2, 4]); // B to be transposed
    const C = new Float64Array([0, 0, 0, 0]);

    dgemm(Transpose.NoTranspose, Transpose.Transpose, m, n, k, 1.0, A, m, B, n, 0.0, C, m);

    // C = I*B^T = B^T = [[1,3], [2,4]]
    expect(C[0]).toBeCloseTo(1);
    expect(C[1]).toBeCloseTo(2);
    expect(C[2]).toBeCloseTo(3);
    expect(C[3]).toBeCloseTo(4);
  });

  test('both transposed: C = A^T*B^T', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 2, 3, 4]); // A = [[1,3], [2,4]]
    const B = new Float64Array([1, 2, 3, 4]); // B = [[1,3], [2,4]]
    const C = new Float64Array([0, 0, 0, 0]);

    dgemm(Transpose.Transpose, Transpose.Transpose, m, n, k, 1.0, A, k, B, n, 0.0, C, m);

    // A^T = [[1,2], [3,4]], B^T = [[1,2], [3,4]]
    // C = A^T*B^T = [[7,10], [15,22]]
    expect(C[0]).toBeCloseTo(7); // 1*1+2*3
    expect(C[1]).toBeCloseTo(15); // 3*1+4*3
    expect(C[2]).toBeCloseTo(10); // 1*2+2*4
    expect(C[3]).toBeCloseTo(22); // 3*2+4*4
  });

  test('with alpha and beta scaling', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 0, 0, 1]); // identity
    const B = new Float64Array([2, 0, 0, 2]); // 2*identity
    const C = new Float64Array([10, 0, 0, 10]); // 10*identity
    const alpha = 3.0,
      beta = 0.5;

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, alpha, A, m, B, k, beta, C, m);

    // C = alpha*A*B + beta*C = 3*I*2I + 0.5*10I = 6I + 5I = 11I
    expect(C[0]).toBeCloseTo(11); // diagonal
    expect(C[1]).toBeCloseTo(0); // off-diagonal
    expect(C[2]).toBeCloseTo(0); // off-diagonal
    expect(C[3]).toBeCloseTo(11); // diagonal
  });

  test('with alpha = 0', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 2, 3, 4]);
    const B = new Float64Array([5, 6, 7, 8]);
    const C = new Float64Array([10, 20, 30, 40]);
    const alpha = 0.0,
      beta = 2.0;

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, alpha, A, m, B, k, beta, C, m);

    // C = 0*A*B + 2*C = 2*[10,20,30,40] = [20,40,60,80]
    expect(C[0]).toBeCloseTo(20);
    expect(C[1]).toBeCloseTo(40);
    expect(C[2]).toBeCloseTo(60);
    expect(C[3]).toBeCloseTo(80);
  });

  test('with beta = 0', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 0, 0, 2]);
    const B = new Float64Array([3, 0, 0, 4]);
    const C = new Float64Array([100, 200, 300, 400]);
    const alpha = 1.0,
      beta = 0.0;

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, alpha, A, m, B, k, beta, C, m);

    // C = A*B + 0*C = [[3,0], [0,8]]
    expect(C[0]).toBeCloseTo(3);
    expect(C[1]).toBeCloseTo(0);
    expect(C[2]).toBeCloseTo(0);
    expect(C[3]).toBeCloseTo(8);
  });

  test('rectangular matrices (m≠n≠k)', () => {
    // A is 3x2, B is 2x4, C should be 3x4
    const m = 3,
      n = 4,
      k = 2;
    const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 3x2 matrix
    const B = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]); // 2x4 matrix
    const C = new Float64Array(m * n).fill(0); // 3x4 matrix

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, m, B, k, 0.0, C, m);

    // First column of C: A * first column of B = A * [1,2]
    expect(C[0]).toBeCloseTo(9); // 1*1 + 4*2 = 9
    expect(C[1]).toBeCloseTo(12); // 2*1 + 5*2 = 12
    expect(C[2]).toBeCloseTo(15); // 3*1 + 6*2 = 15
  });

  test('with different leading dimensions', () => {
    // Store 2x2 matrices in 3x3 spaces with padding
    const m = 2,
      n = 2,
      k = 2;
    const lda = 3,
      ldb = 3,
      ldc = 3;
    const A = new Float64Array([1, 3, 99, 2, 4, 99]); // [[1,2], [3,4]] with padding
    const B = new Float64Array([5, 7, 99, 6, 8, 99]); // [[5,6], [7,8]] with padding
    const C = new Float64Array(ldc * n).fill(0);

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);

    // Should compute as normal 2x2 multiplication, ignoring padding
    expect(C[0]).toBeCloseTo(19); // 1*5 + 2*7
    expect(C[1]).toBeCloseTo(43); // 3*5 + 4*7
    expect(C[3]).toBeCloseTo(22); // 1*6 + 2*8 (skip index 2 due to ldc=3)
    expect(C[4]).toBeCloseTo(50); // 3*6 + 4*8
  });

  test('matrix multiplication is associative: (A*B)*C = A*(B*C)', () => {
    const size = 2;
    const A = new Float64Array([1, 2, 3, 4]);
    const B = new Float64Array([2, 1, 4, 3]);
    const C_matrix = new Float64Array([1, 3, 2, 4]);

    // Compute (A*B)*C
    const AB = new Float64Array(4);
    dgemm(
      Transpose.NoTranspose,
      Transpose.NoTranspose,
      size,
      size,
      size,
      1.0,
      A,
      size,
      B,
      size,
      0.0,
      AB,
      size
    );
    const AB_C = new Float64Array(4);
    dgemm(
      Transpose.NoTranspose,
      Transpose.NoTranspose,
      size,
      size,
      size,
      1.0,
      AB,
      size,
      C_matrix,
      size,
      0.0,
      AB_C,
      size
    );

    // Compute A*(B*C)
    const BC = new Float64Array(4);
    dgemm(
      Transpose.NoTranspose,
      Transpose.NoTranspose,
      size,
      size,
      size,
      1.0,
      B,
      size,
      C_matrix,
      size,
      0.0,
      BC,
      size
    );
    const A_BC = new Float64Array(4);
    dgemm(
      Transpose.NoTranspose,
      Transpose.NoTranspose,
      size,
      size,
      size,
      1.0,
      A,
      size,
      BC,
      size,
      0.0,
      A_BC,
      size
    );

    // Cs should be equal
    for (let i = 0; i < 4; i++) {
      expect(AB_C[i]).toBeCloseTo(A_BC[i], 10);
    }
  });

  test('identity matrix properties', () => {
    const m = 3,
      n = 3,
      k = 3;
    const A = new Float64Array([1, 4, 7, 2, 5, 8, 3, 6, 9]); // arbitrary 3x3
    const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // identity
    const C = new Float64Array(9).fill(0);

    // A * I should equal A
    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, m, I, k, 0.0, C, m);

    for (let i = 0; i < 9; i++) {
      expect(C[i]).toBeCloseTo(A[i]);
    }
  });

  test('throws error for invalid dimensions', () => {
    const A = new Float64Array(4);
    const B = new Float64Array(4);
    const C = new Float64Array(4);

    expect(() =>
      dgemm(Transpose.NoTranspose, Transpose.NoTranspose, -1, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2)
    ).toThrow();
    expect(() =>
      dgemm(Transpose.NoTranspose, Transpose.NoTranspose, 2, -1, 2, 1.0, A, 2, B, 2, 0.0, C, 2)
    ).toThrow();
    expect(() =>
      dgemm(Transpose.NoTranspose, Transpose.NoTranspose, 2, 2, -1, 1.0, A, 2, B, 2, 0.0, C, 2)
    ).toThrow();
  });

  test('modifies original C matrix', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 0, 0, 2]);
    const B = new Float64Array([2, 0, 0, 3]);
    const C = new Float64Array([1, 1, 1, 1]);

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, m, B, k, 1.0, C, m);

    // C = A*B + C = [[2,0], [0,6]] + [[1,1], [1,1]] = [[3,1], [1,7]]
    expect(C[0]).toBeCloseTo(3); // 2+1
    expect(C[1]).toBeCloseTo(1); // 0+1
    expect(C[2]).toBeCloseTo(1); // 0+1
    expect(C[3]).toBeCloseTo(7); // 6+1
  });

  test('handles zero matrices', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([0, 0, 0, 0]);
    const B = new Float64Array([1, 2, 3, 4]);
    const C = new Float64Array([5, 6, 7, 8]);

    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, m, B, k, 2.0, C, m);

    // C = 0*B + 2*C = 2*[5,6,7,8] = [10,12,14,16]
    expect(C[0]).toBeCloseTo(10);
    expect(C[1]).toBeCloseTo(12);
    expect(C[2]).toBeCloseTo(14);
    expect(C[3]).toBeCloseTo(16);
  });

  test('conjugate transpose (C) acts same as transpose (T) for real matrices', () => {
    const m = 2,
      n = 2,
      k = 2;
    const A = new Float64Array([1, 3, 2, 4]);
    const B = new Float64Array([1, 0, 0, 1]);
    const C1 = new Float64Array([0, 0, 0, 0]);
    const C2 = new Float64Array([0, 0, 0, 0]);

    dgemm(Transpose.Transpose, Transpose.NoTranspose, m, n, k, 1.0, A, k, B, k, 0.0, C1, m);
    dgemm(
      Transpose.ConjugateTranspose,
      Transpose.NoTranspose,
      m,
      n,
      k,
      1.0,
      A,
      k,
      B,
      k,
      0.0,
      C2,
      m
    );

    // Cs should be identical for real matrices
    for (let i = 0; i < 4; i++) {
      expect(C1[i]).toBeCloseTo(C2[i]);
    }
  });

  test('performance with larger matrices', () => {
    const m = 50,
      n = 50,
      k = 50;
    const A = new Float64Array(m * k).fill(1.0);
    const B = new Float64Array(k * n).fill(1.0);
    const C = new Float64Array(m * n).fill(0.0);

    const start = performance.now();
    dgemm(Transpose.NoTranspose, Transpose.NoTranspose, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
    const end = performance.now();

    // Each element of C should be k (sum of k ones)
    for (let i = 0; i < m * n; i++) {
      expect(C[i]).toBeCloseTo(k);
    }

    // Should complete in reasonable time (less than 1 second for 50x50)
    expect(end - start).toBeLessThan(1000);
  });
});
