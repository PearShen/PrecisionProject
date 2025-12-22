import torch
import torch.ops
import torch.nn as nn
import torch.nn.functional as F
import types
from functools import wraps
import time
from vllm import _custom_ops as ops
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from utils.data import prepare_input_and_output, transfer_torch2np_data

@dataclass
class OperatorInfo:
    """Information about an operator execution"""
    module: nn.Module = None
    module_name: str = None
    operator_name: str = None
    input_shapes: List[List[int]] = None
    output_shapes: List[List[int]] = None
    input_dtypes: List[str] = None
    output_dtypes: List[str] = None
    inputs: List[np.ndarray] = None
    outputs: List[np.ndarray] = None
    timestamp: float = None

class OperatorCaptureFramework:
    """统一的算子抓取框架"""
    
    def __init__(self, operator_traces=[]):
        self.patched_functions = {}
        self.operator_traces = operator_traces
        self.module_name_dict = {}
        self.ops_count = 0
        self.enabled = True
        
    def enable(self):
        """启用抓取"""
        self.enabled = True
        
    def disable(self):
        """禁用抓取"""
        self.enabled = False
    
    def capture_custom_operator(self, module_path="torch.ops", namespaces_func_names=None):
        """猴子补丁 nn.functional 中的函数"""
        
        # torch.ops._vllm_fa2_C.varlen_fwd,fwd
        
        namespaces_func_names = {
            "_C_cache_ops": [ 
                'reshape_and_cache_flash',
                'reshape_and_cache',
                'flas'
            ],
            "_vllm_fa2_C":[
                "varlen_fwd",
                "fwd",
            ]
        }
        
        
        for namespace, func_names in namespaces_func_names.items():
            module = __import__(module_path, fromlist=[''])
            if not hasattr(module, namespace):
                # 创建命名空间
                namespace_obj = types.ModuleType(f"{module}.{namespace}")
                setattr(module, namespace, namespace_obj)
            
            # 获取命名空间
            module = getattr(module, namespace)
            for func_name in func_names:
                self._wrap_operator(func_name, module)
            
    
    def capture_torch_functional(self, module_path='torch.nn.functional', func_names=None):
        """猴子补丁 nn.functional 中的函数"""
        if func_names is None:
            # 默认抓取常用函数
            func_names = [
                'relu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
                'conv1d', 'conv2d', 'conv3d', 'linear',
                'max_pool1d', 'max_pool2d', 'max_pool3d',
                'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                'dropout', 'batch_norm', 'layer_norm',
                'mse_loss', 'cross_entropy', 'nll_loss',
                'embedding', #'one_hot',
            ]
            func_names = [
                'softmax'
            ]
            func_names = []
        module = __import__(module_path, fromlist=[''])
        for func_name in func_names:
            self._wrap_operator(func_name, module)
    
    def _wrap_operator(self, func_name, module):
        # Convert inputs and outputs to numpy arrays
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            @wraps(original_func)
            def patched_func(*args, **kwargs):
                if not self.enabled:
                    return original_func(*args, **kwargs)
                start_time = time.perf_counter()
                # 执行原始函数
                output = original_func(*args, **kwargs)
                end_time = time.perf_counter()
                
                
                input = list(args)+list(kwargs.values())
                # inputs_np, outputs_np = prepare_input_and_output(input, output)
                
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
                operator_info = OperatorInfo(
                    module=original_func,
                    module_name=func_name,
                    operator_name=func_name,
                    input_shapes=[list(inp.shape) if hasattr(inp, 'shape') else [] for inp in input],
                    input_dtypes=[str(inp.dtype) if hasattr(inp, 'dtype') else str(type(inp).__name__) for inp in input],
                    output_shapes=[list(out.shape) if hasattr(out, 'shape') else [] for out in (outputs_np if isinstance(output, (list, tuple)) else [output])],
                    # output_dtypes=[str(out.dtype) if hasattr(out, 'dtype') else str(type(out).__name__) for out in (outputs_np if isinstance(output, (list, tuple)) else [output])],
                    output_dtypes=[str(out.dtype) if hasattr(out, 'dtype') else str(type(out).__name__) for out in (output if isinstance(output, (list, tuple)) else [output])],
                    inputs=inputs_np,
                    outputs=outputs_np,
                    timestamp=end_time-start_time
                )
                self.operator_traces.append(operator_info)
                return output
            self.patched_functions[func_name] = original_func
            setattr(module, func_name, patched_func)
            self.ops_count +=1
            self.module_name_dict.setdefault(original_func, []).append(func_name)
                
    def restore_functional(self):
        """恢复原始函数"""
        module = __import__('torch.nn.functional', fromlist=[''])
        for func_name, original_func in self.patched_functions.items():
            setattr(module, func_name, original_func)
        self.patched_functions.clear()
        
        self.patched_functions.clear()
        self.operator_traces.clear()
        self.module_name_dict.clear()
        self.ops_count = 0
    
    
        
def test():       
    # 使用示例
    class CustomReshapeAndCache(nn.Module):
        """自定义卷积层"""
        def __init__(self, block_num, block_size, head_num, head_dim, dtype=torch.float16, device="cpu"):
            super().__init__()
            self.kv_cache = torch.randn(2, block_num, block_size, head_num, head_dim, dtype=dtype, device=device)
        
        def forward(self, model_inpt):
            # 这里可能调用自定义卷积算子
            # 假设 custom_conv2d 是自定义算子
            key, value, slotting_map = model_inpt
            print(key.dtype, self.kv_cache[0].dtype)
            
            x = torch.nn.functional.softmax(key, dim=-1)
            ops.reshape_and_cache_flash(
                    key,
                    value,
                    self.kv_cache[0],
                    self.kv_cache[1],
                    slotting_map,
                    "auto",
                    torch.tensor(1.0, dtype=torch.float32).to(key.device),
                    torch.tensor(1.0, dtype=torch.float32).to(key.device),
                )
            return key
    model = CustomReshapeAndCache(block_num=1, block_size=16, head_num=8, head_dim=64, dtype=torch.float16, device="cuda")
        
    # 方法1: 使用 OperatorCaptureFramework
    print("方法1: 使用 OperatorCaptureFramework")
    capture_framework = OperatorCaptureFramework()
    capture_framework.capture_custom_operator()
    capture_framework.capture_torch_functional()

    key = torch.randn(1, 8, 64).to(torch.float16).cuda()
    value = torch.randn(1, 8, 64).to(torch.float16).cuda()
    slotting_mapping = torch.tensor([0,]).to(torch.int64).cuda()
    output = model((key, value, slotting_mapping))
    print(capture_framework.operator_traces)
    
if __name__=="__main__":
    test()