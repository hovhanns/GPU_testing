
import torch
def test_addition():
    device ='cuda'
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([4, 5, 6], device=device)
    assert torch.equal(a + b, torch.tensor([5, 7, 9], device=device))

def test_tensor_multiplication():
    device ='cuda'
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    assert c.is_cuda, "Tensor operation was not executed on the GPU"
    print("Tensor multiplication on GPU successful")