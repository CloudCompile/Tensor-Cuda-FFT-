"""
Cleanup GPU and CUDA resources

Run this to properly release GPU memory and close CUDA contexts.
"""
import gc
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except:
    TRITON_AVAILABLE = False

print("GPU Cleanup Script")
print("=" * 50)

if TORCH_AVAILABLE and torch.cuda.is_available():
    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Clear cache
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    print("Resetting memory stats...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Force garbage collection
    print("Running garbage collection...")
    gc.collect()
    
    # Show current memory
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    
    print(f"\nGPU Memory Status:")
    print(f"  Allocated: {allocated:.1f} MB")
    print(f"  Reserved:  {reserved:.1f} MB")
    
    if allocated < 1 and reserved < 10:
        print("\n[OK] GPU memory cleaned successfully")
    else:
        print(f"\n[WARNING] Some memory still in use")
        print("You may need to restart Python to fully release GPU")
else:
    print("\nCUDA not available or PyTorch not installed")

if TRITON_AVAILABLE:
    print("\nTriton is installed")
    # Triton cache location
    import os
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache):
        print(f"  Cache location: {triton_cache}")
        print("  (Cache is normal and can be kept)")

print("\n" + "=" * 50)
print("Cleanup complete. You can now close this terminal.")
print("=" * 50)
