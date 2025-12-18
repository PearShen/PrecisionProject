import torch

def transfer_torch2np_data(torch_tensor):
    if torch_tensor.dtype == torch.bfloat16:
        return torch_tensor.detach().cpu().view(torch.float16).numpy()
    if torch_tensor.dtype == torch.float8_e4m3fn:
        return torch_tensor.detach().cpu().view(torch.uint8).numpy()
    return torch_tensor.detach().cpu().numpy()