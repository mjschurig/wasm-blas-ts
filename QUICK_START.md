# Quick Start Guide

This guide will help you get started with developing and using wasm-blas-ts.

## For Users

### Installation

```bash
npm install wasm-blas-ts
```

### Basic Usage

```typescript
import { initWasm, daxpy } from 'wasm-blas-ts';

async function example() {
  // Initialize WASM (required once)
  await initWasm();

  // Perform DAXPY: y = alpha * x + y
  const x = new Float64Array([1, 2, 3, 4]);
  const y = new Float64Array([5, 6, 7, 8]);

  daxpy(4, 2.0, x, 1, y, 1);
  console.log(y); // [7, 10, 13, 16]
}

example();
```

## For Developers

### Option 1: Using Dev Container (Recommended)

1. **Prerequisites**
   - Docker installed
   - VS Code with Remote-Containers extension

2. **Steps**

   ```bash
   # Open in VS Code
   code .

   # Click "Reopen in Container" when prompted
   # Everything will be set up automatically!
   ```

3. **Build and Test**
   ```bash
   npm run build
   npm test
   ```

### Option 2: Local Development

1. **Prerequisites**
   - Node.js >= 18
   - Emscripten SDK
   - CMake >= 3.15

2. **Install Emscripten**

   ```bash
   # Clone emsdk
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk

   # Install and activate
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh

   cd ..
   ```

3. **Install Dependencies**

   ```bash
   npm install
   ```

4. **Build**

   ```bash
   # Build WebAssembly
   npm run build:wasm

   # Build TypeScript
   npm run build:ts

   # Or build everything
   npm run build
   ```

5. **Run Tests**

   ```bash
   npm test
   ```

6. **Run Example**
   ```bash
   npm run build
   node -r ts-node/register examples/basic-usage.ts
   ```

## Development Workflow

### Making Changes

1. **Edit C++ code** in `src/cpp/`
2. **Edit TypeScript code** in `src/`
3. **Add tests** in `tests/`
4. **Build and test**
   ```bash
   npm run build
   npm test
   npm run lint
   ```

### Running Linter

```bash
# Check for issues
npm run lint

# Auto-fix issues
npm run lint:fix

# Format code
npm run format
```

### Type Checking

```bash
npm run typecheck
```

### Watch Mode for Tests

```bash
npm run test:watch
```

## Project Structure

```
wasm-blas-ts/
├── src/
│   ├── cpp/          # C++ source code
│   │   ├── daxpy.cpp
│   │   └── blas.h
│   ├── daxpy.ts      # TypeScript wrappers
│   ├── wasm-module.ts # WASM initialization
│   └── index.ts      # Main entry point
├── tests/            # Test files
│   └── daxpy.test.ts
├── examples/         # Usage examples
│   └── basic-usage.ts
├── build/            # WASM build output (generated)
├── dist/             # TypeScript build output (generated)
├── .devcontainer/    # Dev container config
├── .github/          # GitHub Actions workflows
└── CMakeLists.txt    # CMake build config
```

## Common Tasks

### Add a New BLAS Function

1. Create C++ implementation in `src/cpp/`
2. Add export to `CMakeLists.txt`
3. Create TypeScript wrapper in `src/`
4. Add tests in `tests/`
5. Export from `src/index.ts`
6. Update README.md

### Publish to NPM

```bash
# Update version in package.json
npm version patch|minor|major

# This will automatically:
# - Build the project
# - Run tests
# - Run linter
# - Create a git tag

# Push to GitHub
git push && git push --tags

# Create a GitHub release
# The GitHub Action will automatically publish to NPM
```

## Troubleshooting

### Emscripten not found

Make sure to activate Emscripten:

```bash
source /path/to/emsdk/emsdk_env.sh
```

Or add to your shell profile:

```bash
echo 'source /path/to/emsdk/emsdk_env.sh' >> ~/.bashrc
```

### WASM module not loading

Make sure to call `initWasm()` before using any BLAS functions:

```typescript
await initWasm(); // Required!
daxpy(...);       // Now it works
```

### Tests failing

1. Make sure to build first: `npm run build`
2. Check that WASM module was built: `ls build/blas.js`
3. Run with verbose output: `npm test -- --verbose`

## Resources

- [BLAS Reference](http://www.netlib.org/blas/)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Jest Testing](https://jestjs.io/docs/getting-started)
