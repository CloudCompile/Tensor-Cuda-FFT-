# FFT-Tensor - Honest Project Status

## What This Actually Is

An experimental PyTorch extension for storing tensors as sparse frequency coefficients. Achieves 3-10x memory compression at the cost of 25-40% reconstruction error and 10-100x slower operations.

**Not production-ready. Not faster. Not revolutionary. Just an experiment.**

---

## Test Results

**Total:** 46 tests  
**Passing:** 31 (67%)  
**Failing:** 14 (30%)  
**Skipped:** 1 (3%)

### What Works
- Core sparse frequency tensors (15/15 unit tests pass)
- Memory tracking
- Block streaming (reduces memory spikes)
- Basic compression/decompression

### What's Broken
- Convolution theorem matmul (163% error)
- Multi-dimensional convolutions (all failing)
- Frequency attention (dimension bugs)
- Performance claims (actually 100x slower, not faster)

---

## Actual Performance

**Speed:** 10-100x slower than PyTorch (not faster)  
**Compression:** 3-10x (not 100x)  
**Quality:** 25-40% error at useful compression (not <5%)  
**Memory:** Only compresses weights, not activations

---

## Should You Use This?

**No, unless:**
- You're researching frequency-domain representations
- You need to store model checkpoints 5x smaller
- You can tolerate 30%+ quality loss
- Speed doesn't matter

**Better alternatives:**
- INT8 quantization: 4x compression, <0.1% loss, same speed, one line
- Model pruning: 2-10x compression, maintained accuracy
- LoRA: Keep base frozen, train small deltas

---

## What to Believe

**TRUE:**
- Compresses tensors via sparse FFT
- Reduces memory spikes through block streaming
- All unit tests pass

**FALSE:**
- "Revolutionary" - Standard technique since 1960s
- "Zero materialization" - Blocks still decompressed
- "Faster" - Actually 100x slower
- "120B on 6GB" - Ignores activation memory

---

## Installation

```bash
pip install torch numpy
git clone https://github.com/yourusername/fft-tensor.git
cd fft-tensor
python -c "from fft_tensor import sst; print('OK')"
```

---

## Basic Usage

```python
import torch
from fft_tensor import sst

# Compress (works)
weights = torch.randn(1024, 1024)
compressed = sst(weights, sparsity=0.10)
print(f"Compression: {compressed.compress_ratio():.0f}x")  # ~10x

# Decompress (works)
reconstructed = compressed.to_spatial()
error = torch.norm(reconstructed - weights) / torch.norm(weights)
print(f"Error: {error*100:.0f}%")  # ~30%
```

---

## What Needs Fixing

**Critical:**
1. Remove circulant matmul (broken)
2. Remove conv1d/2d/3d (all failing)
3. Fix frequency attention (dimension bugs)
4. Remove performance claims (false)
5. Lower compression expectations (realistic)

**Documentation:**
1. Stop saying "revolutionary"
2. Acknowledge it's slower
3. Report real quality numbers
4. Compare to INT8 honestly

---

## Contributing

Help needed:
1. Fix convolution theorem implementation
2. Benchmark on real models (GPT-2, Llama)
3. Compare quality vs INT8/INT4
4. Tighten test thresholds (currently allow 95% error)

---

## License

MIT - Use at your own risk

---

## FAQ

**Q: Is this faster?**  
A: No. 100x slower.

**Q: Should I use this in production?**  
A: No.

**Q: What about the 100x compression?**  
A: At 1% sparsity with 80%+ error. Useless.

**Q: What's the realistic compression?**  
A: 5-10x with 30-40% error.

**Q: Why should I use this over INT8?**  
A: You shouldn't.

---

**Current Status:** Experimental, mostly broken, needs rewrite

See [TEST_RESULTS.md](TEST_RESULTS.md) for detailed test analysis.
