import torch

def test_pytorch_gpu():
    print("=" * 40)
    print("PyTorch GPU Verification Script")
    print("=" * 40)
    
    # 1. Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ GPU is NOT available to PyTorch.")
        print("Please check your NVIDIA drivers and CUDA toolkit installation.")
        return

    # 2. Get GPU Device Information
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("=" * 40)

    # 3. Perform a quick computation test on the GPU
    try:
        # Set device to GPU
        device = torch.device("cuda")
        
        # Create two random tensors directly on the GPU
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Perform matrix multiplication
        print("Running matrix multiplication on GPU...")
        z = torch.matmul(x, y)
        
        # Sync to make sure operation completes and verify

        torch.cuda.synchronize()
        
        print(f"Result tensor device: {z.device}")
        print("\n✅ Success! Your GPU is working perfectly with PyTorch.")
        
    except Exception as e:
        print(f"\n❌ An error occurred during the GPU test: {e}")

if __name__ == "__main__":
    test_pytorch_gpu()
