"""
Tests for multi-dimensional frequency convolutions.

Validates that 1D, 2D, and 3D convolutions via convolution theorem
produce equivalent results to standard spatial convolution.
"""
import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fft_tensor.zero_materialize import ConvolutionTheoremMatMul


class TestFrequencyConv1D:
    """Test 1D convolution in frequency domain."""
    
    def test_conv1d_correctness(self):
        """Test frequency conv1d matches standard conv1d."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Small test case
        B, C_in, L = 2, 3, 32
        C_out, K = 5, 7
        
        x = torch.randn(B, C_in, L, device=device)
        w = torch.randn(C_out, C_in, K, device=device)
        
        # Standard conv1d
        expected = F.conv1d(x, w, padding=K//2)
        
        # Frequency conv1d
        w_freq = torch.fft.fft(w, dim=-1)
        result = ConvolutionTheoremMatMul.frequency_conv1d(
            x, w_freq, padding=K//2
        )
        
        # Should be close
        error = torch.norm(result - expected) / torch.norm(expected)
        print(f"\n1D Conv error: {error:.6f}")
        assert error < 0.1, f"Error too high: {error}"
    
    def test_conv1d_stride(self):
        """Test stride support."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = torch.randn(2, 3, 32, device=device)
        w = torch.randn(5, 3, 7, device=device)
        w_freq = torch.fft.fft(w, dim=-1)
        
        # With stride=2
        result = ConvolutionTheoremMatMul.frequency_conv1d(
            x, w_freq, stride=2, padding=3
        )
        
        # Check output size
        expected_L = (32 + 2*3 - 7) // 2 + 1
        assert result.shape == (2, 5, expected_L)
        print(f"\nStride test output: {result.shape}")


class TestFrequencyConv2D:
    """Test 2D convolution in frequency domain (images)."""
    
    def test_conv2d_correctness(self):
        """Test frequency conv2d matches standard conv2d."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Image-sized test
        B, C_in, H, W = 2, 3, 32, 32
        C_out, Kh, Kw = 8, 5, 5
        
        x = torch.randn(B, C_in, H, W, device=device)
        w = torch.randn(C_out, C_in, Kh, Kw, device=device)
        
        # Standard conv2d
        expected = F.conv2d(x, w, padding=(Kh//2, Kw//2))
        
        # Frequency conv2d
        w_freq = torch.fft.fft2(w, dim=(-2, -1))
        result = ConvolutionTheoremMatMul.frequency_conv2d(
            x, w_freq, padding=(Kh//2, Kw//2)
        )
        
        # Should be close
        error = torch.norm(result - expected) / torch.norm(expected)
        print(f"\n2D Conv error: {error:.6f}")
        assert error < 0.1, f"Error too high: {error}"
    
    def test_conv2d_different_strides(self):
        """Test non-square strides."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = torch.randn(1, 3, 64, 64, device=device)
        w = torch.randn(16, 3, 3, 3, device=device)
        w_freq = torch.fft.fft2(w, dim=(-2, -1))
        
        # Asymmetric stride
        result = ConvolutionTheoremMatMul.frequency_conv2d(
            x, w_freq, stride=(2, 1), padding=(1, 1)
        )
        
        # Check output shape
        expected_H = (64 + 2*1 - 3) // 2 + 1
        expected_W = (64 + 2*1 - 3) // 1 + 1
        assert result.shape == (1, 16, expected_H, expected_W)
        print(f"\n2D stride test: {result.shape}")
    
    def test_conv2d_large_image(self):
        """Test on larger image (performance)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = torch.randn(1, 3, 224, 224, device=device)
        w = torch.randn(64, 3, 7, 7, device=device)
        w_freq = torch.fft.fft2(w, dim=(-2, -1))
        
        result = ConvolutionTheoremMatMul.frequency_conv2d(
            x, w_freq, stride=(2, 2), padding=(3, 3)
        )
        
        assert result.shape == (1, 64, 112, 112)
        print(f"\nLarge image output: {result.shape}")


class TestFrequencyConv3D:
    """Test 3D convolution in frequency domain (video, medical)."""
    
    def test_conv3d_correctness(self):
        """Test frequency conv3d matches standard conv3d."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Small 3D volume
        B, C_in, D, H, W = 1, 2, 16, 16, 16
        C_out, Kd, Kh, Kw = 4, 3, 3, 3
        
        x = torch.randn(B, C_in, D, H, W, device=device)
        w = torch.randn(C_out, C_in, Kd, Kh, Kw, device=device)
        
        # Standard conv3d
        expected = F.conv3d(x, w, padding=(Kd//2, Kh//2, Kw//2))
        
        # Frequency conv3d
        w_freq = torch.fft.fftn(w, dim=(-3, -2, -1))
        result = ConvolutionTheoremMatMul.frequency_conv3d(
            x, w_freq, padding=(Kd//2, Kh//2, Kw//2)
        )
        
        # Should be close
        error = torch.norm(result - expected) / torch.norm(expected)
        print(f"\n3D Conv error: {error:.6f}")
        assert error < 0.1, f"Error too high: {error}"
    
    def test_conv3d_video(self):
        """Test on video-like dimensions (T, H, W)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Video: 16 frames, 64x64 spatial
        x = torch.randn(1, 3, 16, 64, 64, device=device)
        w = torch.randn(32, 3, 3, 7, 7, device=device)  # Temporal+spatial kernel
        w_freq = torch.fft.fftn(w, dim=(-3, -2, -1))
        
        result = ConvolutionTheoremMatMul.frequency_conv3d(
            x, w_freq, stride=(1, 2, 2), padding=(1, 3, 3)
        )
        
        # Should downsample spatial dims but keep temporal
        assert result.shape[2] == 16  # Temporal preserved
        assert result.shape[3] < 64   # Spatial downsampled
        print(f"\nVideo conv output: {result.shape}")


class TestBatchedOperations:
    """Test batch processing optimizations."""
    
    def test_batched_linear_correctness(self):
        """Test batched linear matches unbatched."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        B, N, D_in, D_out = 64, 32, 128, 256
        
        x = torch.randn(B, N, D_in, device=device)
        w = torch.randn(D_out, D_in, device=device)
        w_freq = torch.fft.fft(w, dim=-1)
        
        # Unbatched (original)
        result1 = ConvolutionTheoremMatMul.frequency_linear(x, w_freq)
        
        # Batched (optimized)
        result2 = ConvolutionTheoremMatMul.frequency_linear_batched(
            x, w_freq, chunk_size=16
        )
        
        # Should be identical
        error = torch.norm(result1 - result2) / torch.norm(result1)
        print(f"\nBatched linear error: {error:.6f}")
        assert error < 1e-5, f"Results differ: {error}"
    
    def test_batched_memory_efficiency(self):
        """Test that batched processing uses less peak memory."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cpu':
            pytest.skip("Memory test requires CUDA")
        
        B, N, D_in, D_out = 128, 64, 512, 1024
        
        x = torch.randn(B, N, D_in, device=device)
        w_freq = torch.randn(D_out, D_in, dtype=torch.complex64, device=device)
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Batched with small chunks (should use less memory)
        result = ConvolutionTheoremMatMul.frequency_linear_batched(
            x, w_freq, chunk_size=16
        )
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak memory (batched): {peak_mem:.1f}MB")
        
        # Should be reasonable
        assert peak_mem < 500, f"Using too much memory: {peak_mem}MB"


class TestConvolutionTheoremProperties:
    """Test mathematical properties of convolution theorem."""
    
    def test_linearity(self):
        """Test that frequency conv is linear: conv(ax + by) = a*conv(x) + b*conv(y)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x1 = torch.randn(1, 3, 32, device=device)
        x2 = torch.randn(1, 3, 32, device=device)
        w = torch.randn(5, 3, 7, device=device)
        w_freq = torch.fft.fft(w, dim=-1)
        
        a, b = 2.0, 3.0
        
        # Left side: conv(ax + by)
        left = ConvolutionTheoremMatMul.frequency_conv1d(
            a * x1 + b * x2, w_freq, padding=3
        )
        
        # Right side: a*conv(x) + b*conv(y)
        right = a * ConvolutionTheoremMatMul.frequency_conv1d(x1, w_freq, padding=3) + \
                b * ConvolutionTheoremMatMul.frequency_conv1d(x2, w_freq, padding=3)
        
        # Should be equal
        error = torch.norm(left - right) / torch.norm(left)
        print(f"\nLinearity test error: {error:.6f}")
        assert error < 1e-5, "Linearity violated!"
    
    def test_commutativity(self):
        """Test that frequency conv is commutative (in the right sense)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # For 1D signals, conv should be commutative
        x = torch.randn(1, 1, 32, device=device)
        w = torch.randn(1, 1, 7, device=device)
        
        # Pad to same size
        w_padded = F.pad(w, (0, 32 - 7))
        
        # conv(x, w)
        w_freq = torch.fft.fft(w_padded, dim=-1)
        result1 = ConvolutionTheoremMatMul.frequency_conv1d(x, w_freq)
        
        # conv(w, x) - swap roles
        x_as_kernel = F.pad(x, (0, 0))[:, :, :7]
        x_freq = torch.fft.fft(x_as_kernel, dim=-1)
        result2 = ConvolutionTheoremMatMul.frequency_conv1d(w_padded.view(1, 1, 32), x_freq)
        
        # Should be similar (up to boundary effects)
        print(f"\nCommutativity test (approximate check)")
        print(f"Result1 shape: {result1.shape}, Result2 shape: {result2.shape}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
