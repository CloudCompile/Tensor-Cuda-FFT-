import torch

data = torch.randn(128, 128)
freq = torch.fft.fft2(data)
mags = torch.abs(freq.flatten())

for sparsity in [0.01, 0.05, 0.10, 0.20, 0.50]:
    k = int(mags.numel() * sparsity)
    vals, idx = torch.topk(mags, k)
    sparse = torch.zeros_like(freq.flatten())
    sparse[idx] = freq.flatten()[idx]
    sparse = sparse.reshape(freq.shape)
    recon = torch.fft.ifft2(sparse).real
    error = torch.norm(recon - data) / torch.norm(data)
    print(f'Sparsity {sparsity*100:5.1f}% -> Error {error*100:6.2f}%')
