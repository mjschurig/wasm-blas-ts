/**
 * Tests for DROT function
 */

import { drot, initWasm } from '../src/index';

describe('DROT - Plane Rotation', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('basic operation with identity rotation (c=1, s=0)', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);
    const c = 1.0;
    const s = 0.0;

    drot(n, x, 1, y, 1, c, s);

    // Identity rotation should leave vectors unchanged
    // [x'] = [1  0] [x] = [x]
    // [y']   [0  1] [y]   [y]
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);

    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(5);
    expect(y[2]).toBeCloseTo(6);
  });

  test('90 degree rotation (c=0, s=1)', () => {
    const n = 2;
    const x = new Float64Array([3, 4]);
    const y = new Float64Array([0, 0]);
    const c = 0.0;
    const s = 1.0;

    drot(n, x, 1, y, 1, c, s);

    // 90° rotation: [x'] = [0  1] [x] = [y]
    //               [y']   [-1 0] [y]   [-x]
    expect(x[0]).toBeCloseTo(0); // y[0]
    expect(x[1]).toBeCloseTo(0); // y[1]
    expect(y[0]).toBeCloseTo(-3); // -x[0]
    expect(y[1]).toBeCloseTo(-4); // -x[1]
  });

  test('45 degree rotation', () => {
    const n = 2;
    const x = new Float64Array([1, 0]);
    const y = new Float64Array([0, 1]);
    const c = Math.cos(Math.PI / 4); // cos(45°) = √2/2
    const s = Math.sin(Math.PI / 4); // sin(45°) = √2/2

    drot(n, x, 1, y, 1, c, s);

    // 45° rotation of unit vectors
    const sqrt2_2 = Math.sqrt(2) / 2;
    expect(x[0]).toBeCloseTo(sqrt2_2); // cos(45°)
    expect(x[1]).toBeCloseTo(sqrt2_2); // sin(45°)
    expect(y[0]).toBeCloseTo(-sqrt2_2); // -sin(45°)
    expect(y[1]).toBeCloseTo(sqrt2_2); // cos(45°)
  });

  test('180 degree rotation (c=-1, s=0)', () => {
    const n = 3;
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);
    const c = -1.0;
    const s = 0.0;

    drot(n, x, 1, y, 1, c, s);

    // 180° rotation: [x'] = [-1  0] [x] = [-x]
    //                [y']   [0  -1] [y]   [-y]
    expect(x[0]).toBeCloseTo(-1);
    expect(x[1]).toBeCloseTo(-2);
    expect(x[2]).toBeCloseTo(-3);

    expect(y[0]).toBeCloseTo(-4);
    expect(y[1]).toBeCloseTo(-5);
    expect(y[2]).toBeCloseTo(-6);
  });

  test('rotation with negative sine', () => {
    const n = 2;
    const x = new Float64Array([1, 0]);
    const y = new Float64Array([0, 1]);
    const c = 0.0;
    const s = -1.0;

    drot(n, x, 1, y, 1, c, s);

    // Rotation with s=-1: [x'] = [0 -1] [x] = [-y]
    //                     [y']   [1  0] [y]   [x]
    expect(x[0]).toBeCloseTo(0); // -y[0] = 0
    expect(x[1]).toBeCloseTo(-1); // -y[1] = -1
    expect(y[0]).toBeCloseTo(1); // x[0] = 1
    expect(y[1]).toBeCloseTo(0); // x[1] = 0
  });

  test('with incx = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 99, 3, 99]);
    const y = new Float64Array([2, 4]);
    const c = 1.0;
    const s = 0.0;

    drot(n, x, 2, y, 1, c, s);

    // Identity rotation with incx=2, effective x = [1, 3]
    expect(x[0]).toBeCloseTo(1); // unchanged
    expect(x[1]).toBeCloseTo(99); // unchanged
    expect(x[2]).toBeCloseTo(3); // unchanged
    expect(x[3]).toBeCloseTo(99); // unchanged

    expect(y[0]).toBeCloseTo(2); // unchanged
    expect(y[1]).toBeCloseTo(4); // unchanged
  });

  test('with incy = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 3]);
    const y = new Float64Array([2, 88, 4, 88]);
    const c = 1.0;
    const s = 0.0;

    drot(n, x, 1, y, 2, c, s);

    // Identity rotation with incy=2, effective y = [2, 4]
    expect(x[0]).toBeCloseTo(1); // unchanged
    expect(x[1]).toBeCloseTo(3); // unchanged

    expect(y[0]).toBeCloseTo(2); // unchanged
    expect(y[1]).toBeCloseTo(88); // unchanged
    expect(y[2]).toBeCloseTo(4); // unchanged
    expect(y[3]).toBeCloseTo(88); // unchanged
  });

  test('with both incx = 2 and incy = 2', () => {
    const n = 2;
    const x = new Float64Array([1, 77, 3, 77]);
    const y = new Float64Array([2, 88, 4, 88]);
    const c = 0.0;
    const s = 1.0;

    drot(n, x, 2, y, 2, c, s);

    // 90° rotation: x' = y, y' = -x
    // effective x = [1, 3], effective y = [2, 4]
    expect(x[0]).toBeCloseTo(2); // y[0]
    expect(x[1]).toBeCloseTo(77); // unchanged
    expect(x[2]).toBeCloseTo(4); // y[2]
    expect(x[3]).toBeCloseTo(77); // unchanged

    expect(y[0]).toBeCloseTo(-1); // -x[0]
    expect(y[1]).toBeCloseTo(88); // unchanged
    expect(y[2]).toBeCloseTo(-3); // -x[2]
    expect(y[3]).toBeCloseTo(88); // unchanged
  });

  test('handles n = 0', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    drot(0, x, 1, y, 1, 0.5, 0.5);

    // Vectors should remain unchanged
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);

    expect(y[0]).toBeCloseTo(4);
    expect(y[1]).toBeCloseTo(5);
    expect(y[2]).toBeCloseTo(6);
  });

  test('handles single element', () => {
    const n = 1;
    const x = new Float64Array([3]);
    const y = new Float64Array([4]);
    const c = 0.6;
    const s = 0.8;

    drot(n, x, 1, y, 1, c, s);

    // [x'] = [0.6  0.8] [3] = [0.6*3 + 0.8*4] = [5.0]
    // [y']   [-0.8 0.6] [4]   [-0.8*3 + 0.6*4]   [0.0]
    expect(x[0]).toBeCloseTo(5.0); // 0.6*3 + 0.8*4
    expect(y[0]).toBeCloseTo(0.0); // -0.8*3 + 0.6*4
  });

  test('rotation preserves vector length', () => {
    const n = 2;
    const x = new Float64Array([3, 4]);
    const y = new Float64Array([0, 0]);
    const c = Math.cos(Math.PI / 4);
    const s = Math.sin(Math.PI / 4);

    // Original combined length of the system
    const originalLength = Math.sqrt(3 * 3 + 4 * 4 + 0 * 0 + 0 * 0); // = 5

    drot(n, x, 1, y, 1, c, s);

    // New combined length of the rotated system
    const newLength = Math.sqrt(x[0] * x[0] + x[1] * x[1] + y[0] * y[0] + y[1] * y[1]);

    expect(newLength).toBeCloseTo(originalLength);
  });

  test('orthogonal rotation matrix property (c² + s² = 1)', () => {
    const n = 2;
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4]);
    const c = 0.8;
    const s = 0.6;

    // Verify orthogonality condition
    expect(c * c + s * s).toBeCloseTo(1.0);

    drot(n, x, 1, y, 1, c, s);

    // Rotation should preserve the sum of squares
    const originalSumSq = 1 * 1 + 2 * 2 + (3 * 3 + 4 * 4);
    const newSumSq = x[0] * x[0] + x[1] * x[1] + (y[0] * y[0] + y[1] * y[1]);

    expect(newSumSq).toBeCloseTo(originalSumSq);
  });

  test('throws error for negative n', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([4, 5, 6]);

    expect(() => drot(-1, x, 1, y, 1, 1.0, 0.0)).toThrow('n must be positive');
  });

  test('throws error for x array too small', () => {
    const x = new Float64Array([1, 2]);
    const y = new Float64Array([3, 4, 5, 6]);

    expect(() => drot(4, x, 1, y, 1, 1.0, 0.0)).toThrow('x array too small');
  });

  test('throws error for y array too small', () => {
    const x = new Float64Array([1, 2, 3, 4]);
    const y = new Float64Array([3, 4]);

    expect(() => drot(4, x, 1, y, 1, 1.0, 0.0)).toThrow('y array too small');
  });

  test('handles fractional rotation parameters', () => {
    const n = 2;
    const x = new Float64Array([1, 0]);
    const y = new Float64Array([0, 1]);
    const c = 1 / Math.sqrt(2);
    const s = 1 / Math.sqrt(2);

    drot(n, x, 1, y, 1, c, s);

    // 45° rotation
    const expected = 1 / Math.sqrt(2);
    expect(x[0]).toBeCloseTo(expected);
    expect(x[1]).toBeCloseTo(expected);
    expect(y[0]).toBeCloseTo(-expected);
    expect(y[1]).toBeCloseTo(expected);
  });

  test('handles large vectors', () => {
    const n = 1000;
    const x = new Float64Array(n).fill(1.0);
    const y = new Float64Array(n).fill(0.0);
    const c = 1.0;
    const s = 0.0;

    drot(n, x, 1, y, 1, c, s);

    // Identity rotation should leave all elements unchanged
    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(1.0);
      expect(y[i]).toBeCloseTo(0.0);
    }
  });

  test('double rotation returns to original (360°)', () => {
    const n = 2;
    const originalX = new Float64Array([1, 2]);
    const originalY = new Float64Array([3, 4]);
    const x = new Float64Array(originalX);
    const y = new Float64Array(originalY);
    const c = Math.cos(Math.PI); // 180°
    const s = Math.sin(Math.PI); // 180°

    // Apply 180° rotation twice (= 360°)
    drot(n, x, 1, y, 1, c, s);
    drot(n, x, 1, y, 1, c, s);

    // Should return to original values
    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(originalX[i], 10);
      expect(y[i]).toBeCloseTo(originalY[i], 10);
    }
  });

  test('composition of rotations', () => {
    const n = 2;
    const x = new Float64Array([1, 0]);
    const y = new Float64Array([0, 1]);

    // First rotation: 45°
    const c1 = Math.cos(Math.PI / 4);
    const s1 = Math.sin(Math.PI / 4);
    drot(n, x, 1, y, 1, c1, s1);

    // Second rotation: another 45° (total = 90°)
    const c2 = Math.cos(Math.PI / 4);
    const s2 = Math.sin(Math.PI / 4);
    drot(n, x, 1, y, 1, c2, s2);

    // Result should be equivalent to 90° rotation
    expect(x[0]).toBeCloseTo(0, 10);
    expect(x[1]).toBeCloseTo(1, 10);
    expect(y[0]).toBeCloseTo(-1, 10);
    expect(y[1]).toBeCloseTo(0, 10);
  });
});
