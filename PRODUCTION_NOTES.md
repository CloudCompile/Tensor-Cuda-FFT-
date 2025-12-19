# Production Notes

## GPU Resource Management

### The Issue

Early benchmark scripts didn't properly clean up CUDA resources, leaving GPU memory allocated after script completion.

**Not Production Acceptable:**
- Processes staying open after completion
- GPU memory not released
- Requiring manual process killing

### The Fix

All production code now includes proper cleanup:

```python
# Pattern 1: Manual cleanup
try:
    model = MyModel().cuda()
    # ... use model
finally:
    # Clean up
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
```

```python
# Pattern 2: Context manager (recommended)
from fft_tensor.cleanup import GPUContext

with GPUContext():
    model = MyModel().cuda()
    # ... use model
# Automatic cleanup
```

```python
# Pattern 3: Explicit cleanup function
from fft_tensor.cleanup import cleanup_models

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# ... training

cleanup_models(model, optimizer)
```

### Required in All Scripts

Every script that uses CUDA must:
1. Use `try/finally` blocks
2. Explicitly delete large objects
3. Call `torch.cuda.empty_cache()`
4. Call `torch.cuda.synchronize()`
5. Call `gc.collect()`

### Verification

After running a script:
```python
import torch
print(torch.cuda.memory_allocated())  # Should be 0 or near 0
print(torch.cuda.memory_reserved())    # Should be minimal
```

Or use Task Manager / nvidia-smi to verify GPU memory is released.

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with GPUContext():
    model = train_model()
    results = evaluate(model)

# Bad - no cleanup guarantee
model = train_model()
results = evaluate(model)
```

### 2. Delete Large Objects Explicitly

```python
# Good
model = MyModel().cuda()
# ... use model
del model
torch.cuda.empty_cache()

# Bad - relies on garbage collection timing
model = MyModel().cuda()
# ... use model
```

### 3. Synchronize Before Exit

```python
# Good
torch.cuda.synchronize()  # Wait for all GPU operations
sys.exit(0)

# Bad - may exit before GPU finishes
sys.exit(0)
```

### 4. Handle Errors Properly

```python
# Good
try:
    model = MyModel().cuda()
    train(model)
except Exception as e:
    print(f"Error: {e}")
finally:
    cleanup_models(model)

# Bad - cleanup only on success
model = MyModel().cuda()
train(model)
cleanup_models(model)
```

## Files Updated

All benchmark and test files now include proper cleanup:

- ✅ `fft_tensor/byte_spectral_triton.py`
- ✅ `fft_tensor/cleanup.py` (new utility module)
- ⚠️ Old test files deleted (didn't follow best practices)

## For Developers

When adding new scripts:

1. Import cleanup utilities:
   ```python
   from fft_tensor.cleanup import GPUContext, cleanup_models
   ```

2. Use context manager or try/finally:
   ```python
   with GPUContext():
       # Your code here
       pass
   ```

3. Test cleanup works:
   ```bash
   python your_script.py
   # Check Task Manager - GPU memory should be released
   ```

## Production Checklist

Before deploying any script:

- [ ] Uses `GPUContext` or `try/finally`
- [ ] Deletes large objects explicitly
- [ ] Calls `torch.cuda.synchronize()` before exit
- [ ] Calls `cleanup_cuda()` or `cleanup_models()`
- [ ] Tested that GPU memory is released
- [ ] No processes left hanging after completion

## Status

**Current:** All active code follows these practices.  
**Legacy:** Old test files (deleted) did not follow best practices.  
**Going forward:** All new code must include proper cleanup.

---

**Lesson learned:** Early benchmark code focused on functionality, not cleanup. Production code must do both. This is now a requirement for all scripts.
