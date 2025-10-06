/**
 * Tests for DROTG function
 */

import { drotg, initWasm } from '../src/index';

describe('DROTG - Givens Rotation Generation', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('classic 3-4-5 triangle', () => {
    const a = 3.0;
    const b = 4.0;

    const result = drotg(a, b);

    // Should generate rotation that makes [3, 4] -> [5, 0]
    expect(result.r).toBeCloseTo(5.0); // sqrt(3² + 4²) = 5
    expect(result.c).toBeCloseTo(0.6); // 3/5
    expect(result.s).toBeCloseTo(0.8); // 4/5

    // Verify the rotation works: [c  s] [a] = [r]
    //                            [-s c] [b]   [0]
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('with a = 0', () => {
    const a = 0.0;
    const b = 5.0;

    const result = drotg(a, b);

    expect(result.r).toBeCloseTo(5.0);
    expect(result.c).toBeCloseTo(0.0);
    expect(Math.abs(result.s)).toBeCloseTo(1.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(Math.abs(rotatedA)).toBeCloseTo(Math.abs(result.r));
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('with b = 0', () => {
    const a = 7.0;
    const b = 0.0;

    const result = drotg(a, b);

    expect(result.r).toBeCloseTo(7.0);
    expect(result.c).toBeCloseTo(1.0);
    expect(result.s).toBeCloseTo(0.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('with both a = 0 and b = 0', () => {
    const a = 0.0;
    const b = 0.0;

    const result = drotg(a, b);

    expect(result.r).toBeCloseTo(0.0);
    expect(result.c).toBeCloseTo(1.0);
    expect(result.s).toBeCloseTo(0.0);
  });

  test('with negative a', () => {
    const a = -3.0;
    const b = 4.0;

    const result = drotg(a, b);

    // The sign of r may vary, but |r| should be sqrt(a² + b²)
    expect(Math.abs(result.r)).toBeCloseTo(5.0);

    // Verify rotation still works
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('with negative b', () => {
    const a = 3.0;
    const b = -4.0;

    const result = drotg(a, b);

    expect(Math.abs(result.r)).toBeCloseTo(5.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('with both negative values', () => {
    const a = -3.0;
    const b = -4.0;

    const result = drotg(a, b);

    expect(Math.abs(result.r)).toBeCloseTo(5.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('orthogonality condition c² + s² = 1', () => {
    const testCases = [
      [3, 4],
      [5, 12],
      [8, 15],
      [7, 24],
      [1, 1],
      [0, 1],
      [1, 0],
    ];

    for (const [a, b] of testCases) {
      const result = drotg(a, b);

      // For non-zero inputs, c² + s² should equal 1
      if (a !== 0 || b !== 0) {
        expect(result.c * result.c + result.s * result.s).toBeCloseTo(1.0, 10);
      }
    }
  });

  test('magnitude preservation', () => {
    const a = 6.0;
    const b = 8.0;

    const result = drotg(a, b);

    const inputMagnitude = Math.sqrt(a * a + b * b);
    expect(Math.abs(result.r)).toBeCloseTo(inputMagnitude);
  });

  test('with very small values', () => {
    const a = 1e-10;
    const b = 1e-10;

    const result = drotg(a, b);

    const expectedR = Math.sqrt(a * a + b * b);
    expect(Math.abs(result.r)).toBeCloseTo(expectedR);

    // Verify rotation still works
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(Math.abs(rotatedB)).toBeLessThan(1e-15); // Should be very close to 0
  });

  test('with very large values', () => {
    const a = 3e10;
    const b = 4e10;

    const result = drotg(a, b);

    expect(Math.abs(result.r)).toBeCloseTo(5e10);
    expect(Math.abs(result.c)).toBeCloseTo(0.6);
    expect(Math.abs(result.s)).toBeCloseTo(0.8);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(Math.abs(rotatedB)).toBeLessThan(1e5); // Should be close to 0 relative to scale
  });

  test('unit vector cases', () => {
    // Test various unit vectors
    const testCases = [
      [1, 0], // x-axis
      [0, 1], // y-axis
      [1 / Math.sqrt(2), 1 / Math.sqrt(2)], // 45° vector
      [-1 / Math.sqrt(2), 1 / Math.sqrt(2)], // 135° vector
    ];

    for (const [a, b] of testCases) {
      const result = drotg(a, b);

      // For unit vectors, |r| should be 1
      expect(Math.abs(result.r)).toBeCloseTo(1.0);

      // Verify rotation works
      const rotatedA = result.c * a + result.s * b;
      const rotatedB = -result.s * a + result.c * b;

      expect(rotatedA).toBeCloseTo(result.r);
      expect(rotatedB).toBeCloseTo(0.0, 10);
    }
  });

  test('fractional values', () => {
    const a = 1.5;
    const b = 2.5;

    const result = drotg(a, b);

    const expectedR = Math.sqrt(1.5 * 1.5 + 2.5 * 2.5);
    expect(Math.abs(result.r)).toBeCloseTo(expectedR);

    // Verify orthogonality
    expect(result.c * result.c + result.s * result.s).toBeCloseTo(1.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('irrational values', () => {
    const a = Math.PI;
    const b = Math.E;

    const result = drotg(a, b);

    const expectedR = Math.sqrt(a * a + b * b);
    expect(Math.abs(result.r)).toBeCloseTo(expectedR);

    // Verify orthogonality
    expect(result.c * result.c + result.s * result.s).toBeCloseTo(1.0);

    // Verify rotation
    const rotatedA = result.c * a + result.s * b;
    const rotatedB = -result.s * a + result.c * b;

    expect(rotatedA).toBeCloseTo(result.r);
    expect(rotatedB).toBeCloseTo(0.0, 10);
  });

  test('consistency with manual calculation', () => {
    const a = 5.0;
    const b = 12.0;

    const result = drotg(a, b);

    // Manual calculation
    const r_manual = Math.sqrt(a * a + b * b); // 13
    const c_manual = a / r_manual; // 5/13
    const s_manual = b / r_manual; // 12/13

    expect(Math.abs(result.r)).toBeCloseTo(r_manual);
    expect(Math.abs(result.c)).toBeCloseTo(Math.abs(c_manual));
    expect(Math.abs(result.s)).toBeCloseTo(Math.abs(s_manual));
  });

  test('generates same rotation for scaled inputs', () => {
    const a1 = 3.0,
      b1 = 4.0;
    const scale = 10.0;
    const a2 = a1 * scale,
      b2 = b1 * scale;

    const result1 = drotg(a1, b1);
    const result2 = drotg(a2, b2);

    // The c and s values should be the same (up to sign)
    expect(Math.abs(result1.c)).toBeCloseTo(Math.abs(result2.c));
    expect(Math.abs(result1.s)).toBeCloseTo(Math.abs(result2.s));

    // The r values should scale proportionally
    expect(Math.abs(result2.r)).toBeCloseTo(Math.abs(result1.r) * scale);
  });
});
