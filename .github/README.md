# GitHub Workflows

This directory contains CI/CD workflows for FFT-Tensor.

## Workflows

### 1. `test-python-fallback.yml` (Main Tests)
**Runs on:** Every push and PR  
**Tests:** PyTorch fallback mode (no CUDA compilation)  
**Python versions:** 3.9, 3.10, 3.11, 3.12  
**Purpose:** Ensure package works without CUDA compilation

**Tests:**
- ✅ Syntax validation
- ✅ Unit tests (15 tests)
- ✅ Import checks
- ✅ Basic functionality
- ✅ Memory management

### 2. `ci.yml` (Lint & Syntax)
**Runs on:** Every push and PR  
**Purpose:** Code quality checks

**Jobs:**
- `lint` - Code formatting (black, flake8)
- `syntax-check` - Python syntax validation
- `examples` - Example file validation

### 3. `documentation.yml` (Docs Check)
**Runs on:** Every push and PR  
**Purpose:** Ensure documentation is complete

**Checks:**
- README.md completeness
- Required documentation files
- Example files exist
- Test files exist

## Why No CUDA Compilation in CI?

CUDA compilation in GitHub Actions requires:
- Self-hosted runner with GPU
- CUDA Toolkit installation (~20GB)
- Visual Studio Build Tools (Windows) or GCC (Linux)
- Long build times (10-20 minutes)

**Solution:** Test PyTorch fallback mode in CI, which:
- ✅ Validates all Python code
- ✅ Tests core functionality
- ✅ Runs quickly (<5 minutes)
- ✅ Works on standard GitHub runners

CUDA compilation can be tested locally or on deployment machines.

## Local Testing

Run the same tests locally:

```bash
# Syntax check
python test_syntax.py

# Unit tests
pytest tests/unit/test_tensor.py -v

# Import test
python -c "from fft_tensor import sst; print('OK')"
```

## Status Badges

Add to README.md:

```markdown
![CI](https://github.com/yourusername/fft-tensor/workflows/Test%20PyTorch%20Fallback%20Mode/badge.svg)
![Docs](https://github.com/yourusername/fft-tensor/workflows/Documentation%20Check/badge.svg)
```
