# Contributing to wasm-blas-ts

Thank you for your interest in contributing to wasm-blas-ts! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/YOUR_USERNAME/wasm-blas-ts.git
   cd wasm-blas-ts
   ```

2. **Install Dependencies**

   ```bash
   npm install
   ```

3. **Set up Emscripten** (if not using Dev Container)
   - Follow instructions at https://emscripten.org/docs/getting_started/downloads.html
   - Or use the provided Dev Container

4. **Build**
   ```bash
   npm run build
   ```

## Development Workflow

1. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following the existing style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests**

   ```bash
   npm test
   npm run lint
   npm run typecheck
   ```

4. **Commit**

   ```bash
   git commit -m "feat: add new feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/)

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Use TypeScript for all source code
- Follow ESLint and Prettier configurations
- Use meaningful variable and function names
- Add JSDoc comments for public APIs
- Keep functions small and focused

## Testing

- Write unit tests for all new functions
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

## Adding New BLAS Functions

1. **Implement C++ version** in `src/cpp/`
2. **Add to CMakeLists.txt** exports
3. **Create TypeScript wrapper** in `src/`
4. **Write comprehensive tests** in `tests/`
5. **Update documentation** in README.md
6. **Export from** `src/index.ts`

## Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Update CHANGELOG.md (if applicable)
- Ensure all CI checks pass
- Respond to review feedback promptly
- Squash commits before merging

## Questions?

Feel free to open an issue for any questions or concerns!
