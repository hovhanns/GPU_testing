import torch

def test_out_of_memory():
    try:
        # Attempt to allocate a large tensor that exceeds GPU memory
        a = torch.randn(100000, 100000, device="cuda")
    except RuntimeError as e:
        assert "out of memory" in str(e).lower(), "Unexpected error"


