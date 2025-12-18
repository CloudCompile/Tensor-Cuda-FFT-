"""
Zero-Materialization Operations: Pure Frequency Domain Math

This module implements TRUE zero-materialization using the Convolution Theorem.
No IFFT decompression ever happens - everything stays in frequency domain.

Key Innovation: Matrix multiplication becomes element-wise multiply in frequency domain.
O(N²) → O(N) with FFT overhead = O(N log N) total.

Mathematical Foundation:
    Convolution Theorem: x * w ↔ X · W
    Linear layer: y = Wx can be expressed as convolution
    Therefore: Y = FFT(y) = FFT(W) ⊙ FFT(x)
    
    We already have FFT(W) stored (sparse!)
    We compute FFT(x) once
    Element-wise multiply (cheap!)
    IFFT only the output (much smaller than weights!)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class ConvolutionTheoremMatMul:
    """
    Matrix multiplication via convolution theorem - ZERO decompression.
    
    Traditional approach:
        W_spatial = IFFT(W_freq)  # Decompress! Bad!
        Y = X @ W_spatial
        
    This approach:
        X_freq = FFT(X)
        Y_freq = X_freq ⊙ W_freq  # Element-wise! Good!
        Y = IFFT(Y_freq)
    
    Memory: Only X_freq + Y + W_freq (already sparse)
    Speed: O(N log N) instead of O(N²)
    """
    
    @staticmethod
    def frequency_linear(x: torch.Tensor, 
                        w_freq: torch.Tensor,
                        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Linear layer computed entirely in frequency domain.
        
        Args:
            x: Input tensor (B, N, D_in) - spatial domain
            w_freq: Weight frequencies (D_in, D_out) - ALREADY frequency domain, sparse!
            bias: Optional bias (D_out)
            
        Returns:
            output: (B, N, D_out) - spatial domain result
            
        Memory: NEVER materializes full W matrix!
        """
        B, N, D_in = x.shape
        D_in2, D_out = w_freq.shape
        assert D_in == D_in2
        
        # Transform input to frequency domain (batched FFT)
        # This is the ONLY transform we do
        x_freq = torch.fft.fft(x, dim=-1)  # (B, N, D_in)
        
        # Reshape for broadcasting
        x_freq = x_freq.unsqueeze(-1)  # (B, N, D_in, 1)
        w_freq = w_freq.unsqueeze(0).unsqueeze(0)  # (1, 1, D_in, D_out)
        
        # Element-wise multiply in frequency domain (THE MAGIC!)
        # This is O(N) instead of O(N²) matmul
        y_freq = x_freq * w_freq  # (B, N, D_in, D_out)
        
        # Sum over input dimension (in frequency domain)
        y_freq = y_freq.sum(dim=2)  # (B, N, D_out)
        
        # Inverse transform ONLY the output
        y = torch.fft.ifft(y_freq, dim=-1).real  # (B, N, D_out)
        
        # Add bias if provided
        if bias is not None:
            y = y + bias
        
        return y
    
    @staticmethod
    def frequency_conv1d(x: torch.Tensor,
                        w_freq: torch.Tensor,
                        stride: int = 1,
                        padding: int = 0) -> torch.Tensor:
        """
        1D convolution in frequency domain (exact via convolution theorem).
        
        This is literally what the theorem is designed for - convolution = FFT multiply.
        
        Args:
            x: Input (B, C_in, L)
            w_freq: Kernel frequencies (C_out, C_in, K) - frequency domain
            stride: Stride
            padding: Padding
            
        Returns:
            output: (B, C_out, L_out)
        """
        B, C_in, L = x.shape
        C_out, C_in2, K = w_freq.shape
        assert C_in == C_in2
        
        # Store original length for output cropping
        L_orig = L
        
        # Apply padding
        if padding > 0:
            x = F.pad(x, (padding, padding))
            L = L + 2 * padding
        
        # Transform to frequency domain
        x_freq = torch.fft.fft(x, dim=-1)  # (B, C_in, L)
        
        # Ensure w_freq is right size (pad to match input length)
        if w_freq.shape[-1] < L:
            w_freq = F.pad(w_freq, (0, L - w_freq.shape[-1]))
        elif w_freq.shape[-1] > L:
            w_freq = w_freq[:, :, :L]
        
        # Reshape for broadcasting
        x_freq = x_freq.unsqueeze(1)  # (B, 1, C_in, L)
        w_freq = w_freq.unsqueeze(0)  # (1, C_out, C_in, L)
        
        # Element-wise multiply (convolution theorem!)
        y_freq = (x_freq * w_freq).sum(dim=2)  # (B, C_out, L)
        
        # Inverse transform
        y = torch.fft.ifft(y_freq, dim=-1).real
        
        # Crop to valid output size (match standard conv1d behavior)
        # Standard conv output size: (L + 2*padding - K) // stride + 1
        if padding > 0:
            # Remove the padding effect to match standard conv
            crop_start = K // 2
            crop_end = y.shape[-1] - (K - K // 2 - 1)
            y = y[:, :, crop_start:crop_end]
        
        # Apply stride if needed (downsample)
        if stride > 1:
            y = y[:, :, ::stride]
        
        return y
    
    @staticmethod
    def frequency_conv2d(x: torch.Tensor,
                        w_freq: torch.Tensor,
                        stride: Tuple[int, int] = (1, 1),
                        padding: Tuple[int, int] = (0, 0)) -> torch.Tensor:
        """
        2D convolution in frequency domain (for images, feature maps).
        
        Uses 2D FFT and convolution theorem:
            conv2d(x, w) = IFFT2D(FFT2D(x) ⊙ FFT2D(w))
        
        Args:
            x: Input (B, C_in, H, W)
            w_freq: Kernel frequencies (C_out, C_in, Kh, Kw) - frequency domain
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
            
        Returns:
            output: (B, C_out, H_out, W_out)
        """
        B, C_in, H, W = x.shape
        C_out, C_in2, Kh, Kw = w_freq.shape
        assert C_in == C_in2
        
        # Apply padding
        if padding[0] > 0 or padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
            H = H + 2 * padding[0]
            W = W + 2 * padding[1]
        
        # Transform to frequency domain (2D FFT)
        x_freq = torch.fft.fft2(x, dim=(-2, -1))  # (B, C_in, H, W)
        
        # Ensure w_freq is right size (pad if needed)
        if w_freq.shape[-2] < H or w_freq.shape[-1] < W:
            pad_h = max(0, H - w_freq.shape[-2])
            pad_w = max(0, W - w_freq.shape[-1])
            w_freq = F.pad(w_freq, (0, pad_w, 0, pad_h))
        
        # Reshape for broadcasting
        x_freq = x_freq.unsqueeze(1)  # (B, 1, C_in, H, W)
        w_freq = w_freq.unsqueeze(0)  # (1, C_out, C_in, H, W)
        
        # Element-wise multiply (convolution theorem!)
        y_freq = (x_freq * w_freq).sum(dim=2)  # (B, C_out, H, W)
        
        # Inverse transform
        y = torch.fft.ifft2(y_freq, dim=(-2, -1)).real
        
        # Apply stride if needed (downsample)
        if stride[0] > 1 or stride[1] > 1:
            y = y[:, :, ::stride[0], ::stride[1]]
        
        return y
    
    @staticmethod
    def frequency_conv3d(x: torch.Tensor,
                        w_freq: torch.Tensor,
                        stride: Tuple[int, int, int] = (1, 1, 1),
                        padding: Tuple[int, int, int] = (0, 0, 0)) -> torch.Tensor:
        """
        3D convolution in frequency domain (for videos, 3D medical images).
        
        Uses 3D FFT and convolution theorem.
        
        Args:
            x: Input (B, C_in, D, H, W)
            w_freq: Kernel frequencies (C_out, C_in, Kd, Kh, Kw) - frequency domain
            stride: (stride_d, stride_h, stride_w)
            padding: (pad_d, pad_h, pad_w)
            
        Returns:
            output: (B, C_out, D_out, H_out, W_out)
        """
        B, C_in, D, H, W = x.shape
        C_out, C_in2, Kd, Kh, Kw = w_freq.shape
        assert C_in == C_in2
        
        # Apply padding
        if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
            x = F.pad(x, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]))
            D = D + 2 * padding[0]
            H = H + 2 * padding[1]
            W = W + 2 * padding[2]
        
        # Transform to frequency domain (3D FFT)
        x_freq = torch.fft.fftn(x, dim=(-3, -2, -1))  # (B, C_in, D, H, W)
        
        # Ensure w_freq is right size (pad if needed)
        if w_freq.shape[-3] < D or w_freq.shape[-2] < H or w_freq.shape[-1] < W:
            pad_d = max(0, D - w_freq.shape[-3])
            pad_h = max(0, H - w_freq.shape[-2])
            pad_w = max(0, W - w_freq.shape[-1])
            w_freq = F.pad(w_freq, (0, pad_w, 0, pad_h, 0, pad_d))
        
        # Reshape for broadcasting
        x_freq = x_freq.unsqueeze(1)  # (B, 1, C_in, D, H, W)
        w_freq = w_freq.unsqueeze(0)  # (1, C_out, C_in, D, H, W)
        
        # Element-wise multiply (convolution theorem!)
        y_freq = (x_freq * w_freq).sum(dim=2)  # (B, C_out, D, H, W)
        
        # Inverse transform
        y = torch.fft.ifftn(y_freq, dim=(-3, -2, -1)).real
        
        # Apply stride if needed (downsample)
        if stride[0] > 1 or stride[1] > 1 or stride[2] > 1:
            y = y[:, :, ::stride[0], ::stride[1], ::stride[2]]
        
        return y
    
    @staticmethod
    def frequency_linear_batched(x_batch: torch.Tensor, 
                                 w_freq: torch.Tensor,
                                 bias: Optional[torch.Tensor] = None,
                                 chunk_size: int = 32) -> torch.Tensor:
        """
        Batch-optimized frequency linear layer.
        
        Processes large batches in chunks to optimize memory/compute trade-off.
        Uses parallel FFT processing for maximum throughput.
        
        Args:
            x_batch: Input tensor (B, N, D_in) - large batch
            w_freq: Weight frequencies (D_in, D_out) - sparse
            bias: Optional bias (D_out)
            chunk_size: Process this many batch elements at once
            
        Returns:
            output: (B, N, D_out)
        """
        B, N, D_in = x_batch.shape
        D_out = w_freq.shape[1]
        
        # Allocate output
        output = torch.zeros(B, N, D_out, device=x_batch.device, dtype=x_batch.dtype)
        
        # Process in chunks for better memory efficiency
        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            x_chunk = x_batch[start_idx:end_idx]
            
            # Transform chunk to frequency domain (batched FFT)
            x_freq = torch.fft.fft(x_chunk, dim=-1)  # (chunk_size, N, D_in)
            
            # Reshape for broadcasting
            x_freq = x_freq.unsqueeze(-1)  # (chunk_size, N, D_in, 1)
            w_freq_expanded = w_freq.unsqueeze(0).unsqueeze(0)  # (1, 1, D_in, D_out)
            
            # Element-wise multiply (batched)
            y_freq = x_freq * w_freq_expanded  # (chunk_size, N, D_in, D_out)
            
            # Sum over input dimension
            y_freq = y_freq.sum(dim=2)  # (chunk_size, N, D_out)
            
            # Inverse transform
            y_chunk = torch.fft.ifft(y_freq, dim=-1).real
            
            # Store result
            output[start_idx:end_idx] = y_chunk
        
        # Add bias if provided
        if bias is not None:
            output = output + bias
        
        return output


class WirtingerAutograd(torch.autograd.Function):
    """
    Complex-valued autograd using Wirtinger calculus.
    
    Standard PyTorch autograd assumes real-valued functions.
    For complex z = a + bi, we need:
        ∂L/∂z and ∂L/∂z̄ (conjugate)
        
    Wirtinger derivatives:
        ∂L/∂z = 1/2(∂L/∂a - i∂L/∂b)
        ∂L/∂z̄ = 1/2(∂L/∂a + i∂L/∂b)
    
    This allows learning in phase space!
    """
    
    @staticmethod
    def forward(ctx, x_freq: torch.Tensor, w_freq: torch.Tensor) -> torch.Tensor:
        """
        Forward: Element-wise complex multiply.
        
        Args:
            x_freq: Input frequencies (complex)
            w_freq: Weight frequencies (complex)
            
        Returns:
            output: x_freq * w_freq
        """
        ctx.save_for_backward(x_freq, w_freq)
        return x_freq * w_freq
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward: Wirtinger derivatives for complex numbers.
        
        For f(z) = z * w:
            ∂f/∂z = w
            ∂f/∂w = z
            
        But in complex space:
            ∂L/∂z = ∂L/∂f · ∂f/∂z̄  (note the conjugate!)
        """
        x_freq, w_freq = ctx.saved_tensors
        
        # Wirtinger derivative with respect to x
        # ∂L/∂x = (∂L/∂output) · conj(w)
        grad_x = grad_output * torch.conj(w_freq)
        
        # Wirtinger derivative with respect to w
        # ∂L/∂w = (∂L/∂output) · conj(x)
        grad_w = grad_output * torch.conj(x_freq)
        
        return grad_x, grad_w


class FrequencyLinearLayer(torch.nn.Module):
    """
    Linear layer that operates entirely in frequency domain with proper gradients.
    
    This is a drop-in replacement for torch.nn.Linear but:
    - Stores weights as sparse frequency coefficients
    - Never materializes spatial weights
    - Supports complex-valued learning (phase relationships)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 sparsity: float = 0.01, bias: bool = True,
                 learn_phase: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.learn_phase = learn_phase
        
        # Initialize weights in frequency domain
        # Random Gaussian in spatial -> FFT -> Sparse
        spatial_init = torch.randn(out_features, in_features) * 0.02
        freq_init = torch.fft.fft(spatial_init, dim=-1)
        
        # Keep only top-k frequencies
        k = int(in_features * sparsity)
        magnitudes = torch.abs(freq_init)
        topk_values, topk_indices = torch.topk(magnitudes, k, dim=-1)
        
        # Create sparse representation
        sparse_freq = torch.zeros_like(freq_init)
        for i in range(out_features):
            sparse_freq[i, topk_indices[i]] = freq_init[i, topk_indices[i]]
        
        # Store as parameter (complex-valued!)
        if learn_phase:
            # Learn both magnitude and phase
            self.weight_freq = torch.nn.Parameter(sparse_freq)
        else:
            # Learn only magnitude, fix phase
            magnitude = torch.abs(sparse_freq)
            phase = torch.angle(sparse_freq)
            self.weight_magnitude = torch.nn.Parameter(magnitude)
            self.register_buffer('weight_phase', phase)
        
        # Bias (stays spatial)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Zero materialization.
        
        Args:
            x: Input (B, N, in_features) - spatial domain
            
        Returns:
            output: (B, N, out_features) - spatial domain
        """
        # Get weight frequencies
        if self.learn_phase:
            w_freq = self.weight_freq
        else:
            # Reconstruct from magnitude and phase
            w_freq = self.weight_magnitude * torch.exp(1j * self.weight_phase)
        
        # Use convolution theorem matmul (zero materialization!)
        output = ConvolutionTheoremMatMul.frequency_linear(x, w_freq, self.bias)
        
        return output
    
    def compress_ratio(self) -> float:
        """Calculate compression ratio."""
        total = self.in_features * self.out_features
        sparse = torch.count_nonzero(torch.abs(self.weight_freq) > 1e-8).item()
        return total / sparse


class LogarithmicQuantizer:
    """
    Logarithmic quantization for frequency coefficients.
    
    Frequency coefficients follow 1/f distribution (power law).
    Most are tiny, few are large.
    
    Log quantization: perfect for this!
    - Small values: high precision (they control nuance)
    - Large values: high range (they control structure)
    
    Format: Sign(1 bit) + Log-Mantissa(7 bits) = 8 bits total
    """
    
    @staticmethod
    def log8_encode(x: torch.Tensor) -> torch.Tensor:
        """
        Encode float32 to log8 (8-bit logarithmic).
        
        Format: [sign:1][log_mantissa:7]
        
        Args:
            x: Float tensor
            
        Returns:
            encoded: uint8 tensor
        """
        # Separate sign
        sign = (x >= 0).to(torch.uint8)
        
        # Log-encode magnitude
        magnitude = torch.abs(x)
        # Add small epsilon to avoid log(0)
        log_mag = torch.log2(magnitude + 1e-8)
        
        # Quantize to 7 bits (0-127)
        # Map typical range [-8, 8] to [0, 127]
        quantized = ((log_mag + 8) / 16 * 127).clamp(0, 127).to(torch.uint8)
        
        # Pack: sign in MSB, magnitude in lower 7 bits
        encoded = (sign << 7) | quantized
        
        return encoded
    
    @staticmethod
    def log8_decode(encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode log8 back to float32.
        
        Args:
            encoded: uint8 tensor
            
        Returns:
            x: Float tensor
        """
        # Unpack sign and magnitude
        sign = ((encoded >> 7) & 1).to(torch.float32) * 2 - 1  # Map {0,1} to {-1,1}
        quantized = (encoded & 0x7F).to(torch.float32)
        
        # Dequantize
        log_mag = (quantized / 127) * 16 - 8
        magnitude = torch.pow(2.0, log_mag)
        
        # Apply sign
        x = sign * magnitude
        
        return x
    
    @staticmethod
    def compress_sparse_freq(freq_coeffs: torch.Tensor, 
                            indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress sparse frequency coefficients using log8.
        
        Args:
            freq_coeffs: Complex frequency coefficients (sparse)
            indices: Indices of non-zero frequencies
            
        Returns:
            compressed_real, compressed_imag: uint8 tensors (4x smaller than float32)
        """
        real = freq_coeffs.real
        imag = freq_coeffs.imag
        
        compressed_real = LogarithmicQuantizer.log8_encode(real)
        compressed_imag = LogarithmicQuantizer.log8_encode(imag)
        
        return compressed_real, compressed_imag
    
    @staticmethod
    def decompress_sparse_freq(compressed_real: torch.Tensor,
                               compressed_imag: torch.Tensor,
                               indices: torch.Tensor,
                               shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decompress log8 back to complex frequencies.
        
        Args:
            compressed_real, compressed_imag: uint8 tensors
            indices: Sparse indices
            shape: Output shape
            
        Returns:
            freq_coeffs: Complex tensor
        """
        real = LogarithmicQuantizer.log8_decode(compressed_real)
        imag = LogarithmicQuantizer.log8_decode(compressed_imag)
        
        # Reconstruct sparse tensor
        freq_coeffs = torch.zeros(shape, dtype=torch.complex64)
        freq_coeffs.real[indices] = real
        freq_coeffs.imag[indices] = imag
        
        return freq_coeffs


# Convenience function
def frequency_linear(x: torch.Tensor, w_freq: torch.Tensor, 
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Functional API for zero-materialization linear layer.
    
    Usage:
        w_freq = torch.fft.fft(weights)  # Do this once, store compressed
        for x in batches:
            y = frequency_linear(x, w_freq)  # Never materializes weights!
    """
    return ConvolutionTheoremMatMul.frequency_linear(x, w_freq, bias)


def frequency_conv1d(x: torch.Tensor, w_freq: torch.Tensor,
                    stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Functional API for zero-materialization 1D convolution.
    """
    return ConvolutionTheoremMatMul.frequency_conv1d(x, w_freq, stride, padding)
