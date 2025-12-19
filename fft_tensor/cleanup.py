"""
Proper cleanup utilities for GPU resources.

Production-quality cleanup that should be called in all scripts.
"""
import torch
import gc


def cleanup_cuda():
    """
    Properly release all CUDA resources.
    
    Call this at the end of every script that uses CUDA.
    """
    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def cleanup_models(*models):
    """
    Properly delete models and free GPU memory.
    
    Usage:
        cleanup_models(model1, model2, optimizer)
    """
    for model in models:
        if model is not None:
            # Move to CPU first
            if hasattr(model, 'cpu'):
                model.cpu()
            
            # Delete
            del model
    
    # Clean CUDA
    cleanup_cuda()


class GPUContext:
    """
    Context manager for GPU operations.
    
    Usage:
        with GPUContext():
            model = MyModel().cuda()
            # ... use model
        # Automatic cleanup when exiting context
    """
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_cuda()
        return False  # Don't suppress exceptions
