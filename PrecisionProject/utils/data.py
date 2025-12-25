import torch
import numpy as np
import json

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

def load_model_config(model_path=None):
    if model_path is None:
        return None
    model_path=get_real_model_path(model_path)
    print(f"===>>>>>begin load model config model_path:{model_path}")
    with open(f"{model_path}/config.json", "r") as read_file:
        model_config = json.load(read_file)
    read_file.close()
    print(f"model_config:{model_config}")
    return model_config

def get_real_model_path(model_path=None):
    if model_path.startswith("/"):
        return model_path
    else:
        # framework auto download model path
        model_download_dir = f"~/.cache/huggingface/hub/models--{model_path.replace('/','--')}/snapshots"
        HASH_PATTERNS = {
            'md5': r'^[a-fA-F0-9]{32}$',          # 32位十六进制
            'sha1': r'^[a-fA-F0-9]{40}$',         # 40位十六进制
            'sha256': r'^[a-fA-F0-9]{64}$',       # 64位十六进制
            'sha512': r'^[a-fA-F0-9]{128}$',      # 128位十六进制
            'sha384': r'^[a-fA-F0-9]{96}$',       # 96位十六进制
            'ripemd160': r'^[a-fA-F0-9]{40}$',    # 40位十六进制
            'blake2b': r'^[a-fA-F0-9]{128}$',     # 128位十六进制
            'blake2s': r'^[a-fA-F0-9]{64}$',      # 64位十六进制
            'crc32': r'^[a-fA-F0-9]{8}$',         # 8位十六进制
            'adler32': r'^[a-fA-F0-9]{8}$',       # 8位十六进制
            
            # 通用 Hash 模式
            'hex_8_32': r'^[a-fA-F0-9]{8,32}$',   # 8-32位十六进制
            'hex_32_128': r'^[a-fA-F0-9]{32,128}$', # 32-128位十六进制
            'any_hex': r'^[a-fA-F0-9]+$',         # 任意长度十六进制
        }
        import os
        import re
        model_download_dir = os.path.expanduser(model_download_dir)
        for hash_type in HASH_PATTERNS:
            pattern = re.compile(HASH_PATTERNS[hash_type])
            for item in os.listdir(model_download_dir):
                if pattern.match(item):
                    return f"{model_download_dir}/{item}/"
        raise "no find model config path!"