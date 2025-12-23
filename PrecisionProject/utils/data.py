import torch
import numpy as np

def transfer_torch2np_data(torch_tensor):
    if torch_tensor.dtype == torch.bfloat16:
        return torch_tensor.detach().cpu().view(torch.float16).numpy()
    if torch_tensor.dtype == torch.float8_e4m3fn:
        return torch_tensor.detach().cpu().view(torch.uint8).numpy()
    return torch_tensor.detach().cpu().numpy()

def prepare_input_and_output(self, input, output):
    inputs_np = []
    for inp in input:
        if isinstance(inp, torch.Tensor):
            inputs_np.append(transfer_torch2np_data(inp))
        else:
            inputs_np.append(np.array(inp))

    outputs_np = []
    if isinstance(output, torch.Tensor):
        outputs_np.append(transfer_torch2np_data(output))
    elif isinstance(output, (list, tuple)):
        for out in output:
            if isinstance(out, torch.Tensor):
                outputs_np.append(transfer_torch2np_data(out))
            else:
                outputs_np.append(np.array(out))
    else:
        outputs_np.append(np.array(output))
    return inputs_np, outputs_np