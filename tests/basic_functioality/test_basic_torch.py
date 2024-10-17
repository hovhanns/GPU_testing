import torch

def test_availability():
    assert torch.cuda.is_available() == True
