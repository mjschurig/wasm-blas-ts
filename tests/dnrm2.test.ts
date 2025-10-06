/**
 * Tests for DNRM2 function
 */

import { dnrm2, initWasm } from '../src/index';

describe('DNRM2 - Euclidean Norm', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with incx=1', () => {
    const n = 2;
    const x = new Float64Array([3, 4]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    expect(result).toBeCloseTo(5.0);
  });

  test('with single element', () => {
    const n = 1;
    const x = new Float64Array([7]);

    const result = dnrm2(n, x, 1);

    // norm = |7| = 7
    expect(result).toBeCloseTo(7.0);
  });

  test('with negative values', () => {
    const n = 2;
    const x = new Float64Array([-3, -4]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5
    expect(result).toBeCloseTo(5.0);
  });

  test('with mixed positive and negative values', () => {
    const n = 3;
    const x = new Float64Array([1, -2, 2]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt(1^2 + (-2)^2 + 2^2) = sqrt(1 + 4 + 4) = sqrt(9) = 3
    expect(result).toBeCloseTo(3.0);
  });

  test('with zero vector', () => {
    const n = 4;
    const x = new Float64Array([0, 0, 0, 0]);

    const result = dnrm2(n, x, 1);

    expect(result).toBeCloseTo(0.0);
  });

  test('with one non-zero element', () => {
    const n = 4;
    const x = new Float64Array([0, 0, 5, 0]);

    const result = dnrm2(n, x, 1);

    expect(result).toBeCloseTo(5.0);
  });

  test('with incx = 2', () => {
    const n = 3;
    const x = new Float64Array([1, 99, 2, 99, 2, 99]);

    const result = dnrm2(n, x, 2);

    // effective x = [1, 2, 2] (at indices 0, 2, 4)
    // norm = sqrt(1^2 + 2^2 + 2^2) = sqrt(1 + 4 + 4) = sqrt(9) = 3
    expect(result).toBeCloseTo(3.0);
  });

  test('with incx = 3', () => {
    const n = 2;
    const x = new Float64Array([6, 0, 0, 8, 0, 0]);

    const result = dnrm2(n, x, 3);

    // effective x = [6, 8] (at indices 0, 3)
    // norm = sqrt(6^2 + 8^2) = sqrt(36 + 64) = sqrt(100) = 10
    expect(result).toBeCloseTo(10.0);
  });

  test('with negative incx', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 2]);

    const result = dnrm2(n, x, -1);

    // Should compute the same norm regardless of direction
    // norm = sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
    expect(result).toBeCloseTo(3.0);
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);

    const result = dnrm2(0, x, 1);

    expect(result).toBe(0.0);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(1.0);

    const result = dnrm2(n, x, 1);

    // norm = sqrt(1000 * 1^2) = sqrt(1000) ≈ 31.62
    expect(result).toBeCloseTo(Math.sqrt(1000));
  });

  test('handles very small values', () => {
    const n = 2;
    const x = new Float64Array([3e-5, 4e-5]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt((3e-5)^2 + (4e-5)^2) = sqrt(9e-10 + 16e-10) = sqrt(25e-10) = 5e-5
    expect(result).toBeCloseTo(5e-5);
  });

  test('handles very large values', () => {
    const n = 2;
    const x = new Float64Array([3e5, 4e5]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt((3e5)^2 + (4e5)^2) = 5e5
    expect(result).toBeCloseTo(5e5);
  });

  test('orthogonal unit vectors', () => {
    const n = 3;
    const x = new Float64Array([1, 0, 0]);

    const result = dnrm2(n, x, 1);

    expect(result).toBeCloseTo(1.0);
  });

  test('unit vector in different direction', () => {
    const n = 3;
    const x = new Float64Array([0, 0, 1]);

    const result = dnrm2(n, x, 1);

    expect(result).toBeCloseTo(1.0);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dnrm2(-1, x, 1)).toThrow('n must be positive');
  });

  test('throws error for array too small', () => {
    const x = new Float64Array([1, 2]);

    expect(() => dnrm2(4, x, 1)).toThrow('x array too small');
  });

  test('throws error for array too small with incx > 1', () => {
    const x = new Float64Array([1, 2, 3]);

    expect(() => dnrm2(3, x, 2)).toThrow('x array too small');
  });

  test('handles fractional values', () => {
    const n = 2;
    const x = new Float64Array([1.5, 2.0]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt(1.5^2 + 2.0^2) = sqrt(2.25 + 4) = sqrt(6.25) = 2.5
    expect(result).toBeCloseTo(2.5);
  });

  test('normalized vector should have norm 1', () => {
    const n = 3;
    // Create a vector and normalize it
    const originalVector = [1, 2, 3];
    const originalNorm = Math.sqrt(1 + 4 + 9); // sqrt(14)
    const x = new Float64Array(originalVector.map((val) => val / originalNorm));

    const result = dnrm2(n, x, 1);

    expect(result).toBeCloseTo(1.0, 10); // High precision check
  });

  test('handles precision with irrational numbers', () => {
    const n = 2;
    const x = new Float64Array([Math.PI, Math.E]);

    const result = dnrm2(n, x, 1);

    const expected = Math.sqrt(Math.PI * Math.PI + Math.E * Math.E);
    expect(result).toBeCloseTo(expected);
  });

  test('handles unrolled loop (larger n)', () => {
    const n = 8;
    const x = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);

    const result = dnrm2(n, x, 1);

    // norm = sqrt(8 * 1^2) = sqrt(8) = 2*sqrt(2)
    expect(result).toBeCloseTo(Math.sqrt(8));
  });

  test('scale invariance test', () => {
    const n = 3;
    const scale = 2.0;
    const x = new Float64Array([1, 2, 3]);
    const scaledX = new Float64Array([2, 4, 6]);

    const result1 = dnrm2(n, x, 1);
    const result2 = dnrm2(n, scaledX, 1);

    // ||scale * x|| = scale * ||x||
    expect(result2).toBeCloseTo(scale * result1);
  });
});
