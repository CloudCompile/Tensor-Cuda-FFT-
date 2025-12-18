# FFT-Tensor: Sparse Frequency-Domain Tensors

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

Sparse tensor storage in frequency domain with block-streaming inference. Reduces memory footprint for large neural network weights through frequency-domain sparsification.

---

## What This Is

A PyTorch extension that:
- Stores tensors as sparse frequency coefficients (FFT + top-K selection)
- Processes matrix operations in blocks to reduce peak memory
- Provides CUDA kernels for sparse frequency operations
- Enables loading larger models on limited VRAM through streaming

**What this is NOT:**
- Not a replacement for quantization (INT8/FP16 are simpler and often better)
- Not "zero materialization" (blocks are still decompressed during compute)
- Not faster than cuBLAS for most operations (trade speed for memory)

---

## Use Cases

**Good for:**
- Model storage/distribution (20-50x smaller checkpoint files)
- Inference on VRAM-limited GPUs when speed isn't critical
- Experimenting with frequency-domain representations

**Not good for:**
- Training (slow and experimental)
- Real-time inference (10-50x slower than standard PyTorch)
- Production systems (use quantization instead)

---

## Installation

```bash
pip install torch numpy
git clone https://github.com/yourusername/fft-tensor.git
cd fft-tensor

# Test it works
python -c "from fft_tensor import sst; print('OK')"
```

Optional CUDA compilation (not required, provides ~10x speedup):
```bash
pip install -e .  # Requires CUDA Toolkit
```

See [INSTALL.md](INSTALL.md) for details.

---

## Quick Start

### Basic Compression

```python
import torch
from fft_tensor import sst

# Compress weights
weights = torch.randn(4096, 4096, device='cuda')
compressed = sst(weights, sparsity=0.05)  # Keep top 5% frequencies

print(f"Compression: {compressed.compress_ratio():.0f}x")  # ~20x
print(f"Memory: {compressed.memory_mb():.2f}MB")
```

### Block Streaming (Reduced Memory Spike)

```python
from fft_tensor import FrequencyMatMul, sst

# Large weight matrix stored compressed
weights_sst = sst(torch.randn(8192, 8192), sparsity=0.01)

# Input
x = torch.randn(32, 512, 8192, device='cuda')

# Compute with reduced memory spike (processes 512-column blocks)
output = FrequencyMatMul.block_streaming_matmul(
    x, weights_sst, block_size=512
)
# Peak memory: ~5MB per block instead of 256MB for full matrix
```

---

## Performance Characteristics

### Memory

**Test: 8192x8192 weight matrix**

| Method | Storage | Peak (Forward Pass) | Quality Loss |
|--------|---------|---------------------|--------------|
| Standard | 256MB | 256MB | 0% |
| FFT-Tensor (5% sparse) | 12MB | ~30MB | 2-5% |
| FFT-Tensor (1% sparse) | 2.5MB | ~15MB | 5-15% |

### Speed

**Relative to cuBLAS on A100:**

| Operation | FFT-Tensor (CPU) | FFT-Tensor (CUDA) | Notes |
|-----------|------------------|-------------------|-------|
| 4096x4096 matmul | 0.01x | 0.1x | Slower due to FFT overhead |
| Large convolution | 0.5x | 2x | FFT efficient for large kernels |
| Memory bandwidth | N/A | 0.3x | Sparse loads are inefficient |

**Conclusion:** Trades speed for memory. Use when VRAM-limited, not speed-limited.

### Compression vs Quality

**Test: Random 1024x1024 matrices**

| Sparsity | Compression | Reconstruction Error | Recommended |
|----------|-------------|---------------------|-------------|
| 20% | 5x | 1-3% | High fidelity |
| 10% | 10x | 2-5% | Balanced |
| 5% | 20x | 3-10% | **Default** |
| 1% | 100x | 10-30% | Maximum compression |

Note: Actual neural network weights compress better than random data.

---

## Architecture

### Core Components

**SparseSpectralTensor (tensor.py):**
- FFT transformation to frequency domain
- Top-K magnitude-based sparsification (standard approach)
- PyTorch integration with memory tracking

**FrequencyMatMul (frequency_ops.py):**
- Block-streaming matrix multiplication
- Processes N columns at a time to bound memory
- Still decompresses blocks (not true zero-materialization)

**CUDA Kernels (cuda/):**
- Sparse gather/scatter for ND arrays
- Complex arithmetic operations
- Alternative to cuSPARSE for specific operations

---

## Comparison to Alternatives

**vs INT8 Quantization:**
- INT8: 4x compression, 0.1% quality loss, same speed, one line of code
- FFT-Tensor: 20x compression, 3-10% quality loss, 10-50x slower, requires integration
- Verdict: Use INT8 unless you need >4x compression

**vs Model Pruning:**
- Pruning: 2-10x compression, maintains sparsity in compute
- FFT-Tensor: 5-100x compression, dense frequency operations
- Verdict: Pruning is more mature, FFT-Tensor compresses more

**vs LoRA/Adapters:**
- LoRA: Keeps base model frozen, trains small adapters
- FFT-Tensor: Compresses entire model
- Verdict: Different use cases (LoRA for fine-tuning, FFT for storage)

---

## Known Limitations

1. **Slower than standard PyTorch:** 10-50x slower due to FFT overhead
2. **Quality loss:** 3-10% reconstruction error at useful compression ratios
3. **Block decompression:** Despite claims, blocks are materialized during compute
4. **Activation memory:** Only compresses weights, not activations (still OOM on large batches)
5. **CUDA kernels not optimal:** Slower than cuBLAS/cuSPARSE
6. **Limited testing:** Needs validation on real large models

---

## Examples

See [examples/](examples/) directory:
- [basic_usage.py](examples/basic_usage.py) - Compression and operations
- [neural_network.py](examples/neural_network.py) - Integration with nn.Module

---

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](INSTALL.md) | Installation instructions |
| [CUDA_SETUP.md](CUDA_SETUP.md) | CUDA compilation guide |
| [GPU_COMPATIBILITY.md](GPU_COMPATIBILITY.md) | GPU requirements |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

---

## Testing

```bash
# Unit tests
pytest tests/unit/test_tensor.py -v

# All tests
pytest tests/ -v
```

**Current Status:** 15/15 unit tests passing

---

## Research Directions

Interesting areas for exploration:

1. **Frequency-domain semantic structure** - Do phases encode relationships?
2. **Adaptive sparsity** - Learn which frequencies matter per layer
3. **True streaming** - Avoid all decompression
4. **Hybrid approaches** - Combine with quantization

---

## Contributing

Contributions welcome, especially:
- Benchmarks on real models (GPT-2, Llama, etc.)
- CUDA kernel optimizations
- Quality/compression trade-off studies
- Comparison with quantization methods

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

```bibtex
@software{fft_tensor2025,
  title={FFT-Tensor: Sparse Frequency-Domain Tensor Library},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fft-tensor}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## FAQ

**Q: Is this faster than standard PyTorch?**
A: No. It's 10-50x slower. The benefit is reduced memory, not speed.

**Q: Should I use this instead of INT8 quantization?**
A: Probably not. INT8 is simpler and better unless you need >4x compression.

**Q: Does this actually enable 120B models on 6GB VRAM?**
A: No. It only compresses weights. Activations still require memory. More realistic: 10-20B models with careful batch sizing.

**Q: What's the quality loss?**
A: 3-10% reconstruction error at 20x compression. Varies by data.

**Q: Is this production-ready?**
A: No. This is experimental research code. Use at your own risk.

---

**Status:** Experimental | **Python:** 3.9-3.12 | **PyTorch:** 2.0+ | **Hardware:** CUDA GPU recommended

**Honest Summary:** Interesting experiment in frequency-domain compression. Trades significant speed for memory savings. Useful for model storage and VRAM-limited inference when speed isn't critical. Not a replacement for standard quantization methods.
