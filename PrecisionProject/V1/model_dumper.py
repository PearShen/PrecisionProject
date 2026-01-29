"""
Model structure and operator execution dumper for PyTorch and vLLM frameworks.
Enhanced with comprehensive operator capture capabilities.
"""

import torch
import torch.nn as nn
import json
import time
from typing import Any, Dict, List
import shutil
import os
import numpy as np
import pandas as pd
from functools import wraps
import types
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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
    # inputs: List[np.ndarray] = None
    # outputs: List[np.ndarray] = None
    start_timestamp: float = None
    end_timestamp: float = None
    duration_time: float = None
    attr: dict =None

class ModelDumper:
    """Dumps model structure and operator execution traces"""

    def __init__(self, save_path, model, model_path=None, dump_start_iter=0, dump_iter_count=3, enable_ops_eff: bool = True):
        """
        Initialize the model dumper.

        Args:
        """
        self.save_path = save_path
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        self.dump_node_count=0
        self.iteration_count = 0
        self.dump_start_iter=dump_start_iter
        self.dump_iter_count=dump_iter_count
        self.curr_dump_iter = dump_start_iter - 1
        self.pre_operator_traces = []
        self.pre_hooks = []
        self.hooks = []
        self.transformer_ops_head = None
        self.transformer_ops_tail = None
        self.module_name_dict = {}
        self.hook_registry = set()
        self.ops_count=0
        self.patched_functions = {}
        self.operator_infos = []
        
        # 自定义算子和nn functional part
        self.namespaces_func_names = {
            
            # cuda
            "torch.ops._C_cache_ops": [ 
                'reshape_and_cache_flash',
                'reshape_and_cache',
                'flas'
            ],
            # cuda
            "torch.ops._vllm_fa2_C":[
                "varlen_fwd",
                "fwd",
            ],
            
            # Phytium npu
            "torch.ops.phy": [
                "reshape_and_cache",
                "concat_and_cache_mla",
                "phy_flash_attention",
                "decode_attention",
                # "phy_attention_mask_generation",
            ],
            
            "torch.nn.functional": [
                'softmax'
                # 'relu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
                # 'conv1d', 'conv2d', 'conv3d', 'linear',
                # 'max_pool1d', 'max_pool2d', 'max_pool3d',
                # 'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                # 'dropout', 'batch_norm', 'layer_norm',
                # 'mse_loss', 'cross_entropy', 'nll_loss',
                # 'embedding', #'one_hot',
            ],
            
        }
        
        self.attr_ops = [
            "varlen_fwd",
            "fwd",
        ]
        
        self._register_torch_hooks(model=model, model_name="")
        self._capture_custom_functional_operator(self.namespaces_func_names)
        self.parent_module_name = self.transformer_ops_tail
        self.enabled = False
        self.model_path=model_path
        self.enable_ops_eff =enable_ops_eff and (model_path is not None)
        if self.enable_ops_eff:
            model_config = load_model_config(model_path)
            self.ops_eff_manager = ModelEfficienyTransformer(model_config)
        
    def enable(self):
        """启用抓取"""
        self.enabled = True
        
    def disable(self):
        """禁用抓取"""
        self.enabled = False       
            
    def _should_register_hook(self, module):
        """判断是否需要注册钩子"""
        module_id = id(module)
        if module_id in self.hook_registry:
            return False
        return True
    
    def pre_hook_fn(self, name, module, input):
        # Convert inputs and outputs to numpy arrays
        # inputs_np, outputs_np = prepare_input_and_output(input, output)
        
        if not self.enabled:
            return input
            
        inputs_np = []

        pre_operator_info_dict = obj = dict(
            # iter=iter_cout,
            # ops_idx=iter_ops_idx,
            module_name=name,     #data.module_name,
            operator_name=type(module).__name__,
            input_shapes=[list(inp.shape) if hasattr(inp, 'shape') else [] for inp in input],
            input_dtypes=[str(inp.dtype) if hasattr(inp, 'dtype') else str(type(inp).__name__) for inp in input],
            start_timestamp=time.perf_counter(),
        )
        self.pre_operator_traces.append(pre_operator_info_dict)
        return input
    
    def _get_iter_idx(self, module_name, dump_node_count):
        if (module_name == self.transformer_ops_head and self.parent_module_name == self.transformer_ops_tail):
            self.curr_dump_iter += 1
            self.dump_node_count = 0
        return self.curr_dump_iter, self.dump_node_count 

    def hook_fn(self, name, module, input, output):
        # Convert inputs and outputs to numpy arrays
        if not self.enabled:
            return output
        timestamp=time.perf_counter()
        if name == self.transformer_ops_head:
            self.iteration_count += 1
        if self.iteration_count <= self.dump_start_iter or self.iteration_count > (self.dump_start_iter+self.dump_iter_count):
            return
        curr_dump_iter, iter_inner_idx = self._get_iter_idx(name, self.dump_node_count)
        inp_shape, inp_dtype, oup_shape, oup_dtype = self._dump_node_data(self.save_path, input, output, name, curr_dump_iter, iter_inner_idx)
        # 使用示例
        operator_info_dict = dict(
            iter=curr_dump_iter,
            ops_idx=iter_inner_idx,
            module_name=name,     #data.module_name,
            operator_name=type(module).__name__,
            input_shapes=inp_shape,
            input_dtypes=inp_dtype,
            output_shapes=oup_shape,
            output_dtypes=oup_dtype,
            start_timestamp=self.pre_operator_traces[-1]["start_timestamp"],
            end_timestamp=timestamp,
            duration_time=timestamp-self.pre_operator_traces[-1]["start_timestamp"]
        )
        self.append_json_line(f"{self.save_path}/operator_info.json",operator_info_dict)
        self.dump_node_count += 1
        return output

    def _register_hook_with_deduplication(self, child, pre_hook_fn, hook_fn, full_name=''):
        """去重注册钩子"""
        if self._should_register_hook(child) or "lm_head" in full_name:
            
            # start time hook
            pre_hook = child.register_forward_pre_hook(
                lambda m, inp, name=full_name: pre_hook_fn(name, m, inp)
            )
            
            # 为每个子模块注册钩子
            hook = child.register_forward_hook(
                lambda m, inp, out, name=full_name: hook_fn(name, m, inp, out)
            )
            self.pre_hooks.append(pre_hook)
            self.hooks.append(hook)
            self.hook_registry.add(id(child))

        self.module_name_dict.setdefault(child, []).append(full_name)
        if self.transformer_ops_head is None:
            self.transformer_ops_head = full_name
        self.transformer_ops_tail = full_name
        self.ops_count += 1
    
    def _register_torch_hooks(self, model: nn.Module, model_name: str) -> None:
        """Register forward hooks for PyTorch model"""
        
        # Register hooks for all layers
        for name, child in model.named_children():
            
            full_name = f"{model_name}.{name}" if model_name else name
            if len(list(child.named_children())) == 0:
                self._register_hook_with_deduplication(
                    child, self.pre_hook_fn, self.hook_fn, full_name
                )
            else:
                # 递归注册
                self._register_torch_hooks(child, model_name=full_name)
    
    def _dump_node_data(self, path, input_tensors, output_tensors, prefix, curr_dump_iter, iter_inner_idx):
        inp_cnt = 0
        inp_shape = []
        inp_dtype = []
        if isinstance(input_tensors, torch.Tensor):
            self._dump_tensor(path, input_tensors, f"iter{curr_dump_iter}_ops{iter_inner_idx}_{prefix}_input{inp_cnt}")
            inp_shape.append(list(input_tensors.shape))
            inp_dtype.append(str(input_tensors.dtype))
            inp_cnt += 1
        elif input_tensors is not None:
            for inp in input_tensors:
                if not isinstance(inp, torch.Tensor):
                    continue
                self._dump_tensor(path, inp, f"iter{curr_dump_iter}_ops{iter_inner_idx}_{prefix}_input{inp_cnt}")
                inp_shape.append(list(inp.shape))
                inp_dtype.append(str(inp.dtype))
                inp_cnt += 1
        oup_cnt = 0
        oup_shape = []
        oup_dtype = []
        if isinstance(output_tensors, torch.Tensor):
            self._dump_tensor(path, output_tensors, f"iter{curr_dump_iter}_ops{iter_inner_idx}_{prefix}_output{oup_cnt}")
            oup_shape.append(list(output_tensors.shape))
            oup_dtype.append(str(output_tensors.dtype))
            oup_cnt += 1
        elif output_tensors is not None:
            for oup in output_tensors:
                if not isinstance(oup, torch.Tensor):
                    continue
                self._dump_tensor(path, oup, f"iter{curr_dump_iter}_ops{iter_inner_idx}_{prefix}_output{oup_cnt}")
                oup_shape.append(list(oup.shape))
                oup_dtype.append(str(oup.dtype))
                oup_cnt += 1
            
        return inp_shape, inp_dtype, oup_shape, oup_dtype
            
    def _dump_tensor(self, path, torch_tensor, name):
        if not os.path.exists(path):
            os.makedirs(path)
        if torch_tensor.ndim >=4:
            np.array([]).tofile(f"{path}/{name}.bin")
            return
        if torch_tensor.dtype == torch.bfloat16:
            torch_tensor.detach().cpu().view(torch.float16).numpy().tofile(f"{path}/{name}.bin")
        elif torch_tensor.dtype == torch.float8_e4m3fn:

            torch_tensor.detach().cpu().view(torch.uint8).numpy().tofile(f"{path}/{name}.bin")
        else:
            torch_tensor.detach().cpu().numpy().tofile(f"{path}/{name}.bin")
            
    def append_json_line(self, filename, data):
        """
        以JSON Lines格式追加数据（每行一个JSON对象）
        """
        if self.enable_ops_eff:
            ops_type, computes_ops, memory_byte, computes_efficiency, memory_efficiency = self.ops_eff_manager.get_ops_efficiency(data)
            data["ops_type"] = ops_type
            data["computes_ops"] = float(computes_ops)
            data["memory_byte"] = int(memory_byte)
            data["computes_efficiency"] = computes_efficiency
            data["memory_efficiency"] = memory_efficiency
        with open(filename, 'a', encoding='utf-8') as file:
            json_str = json.dumps(data, ensure_ascii=False)
            file.write(json_str + '\n')
        self.operator_infos.append(data)
        
            
    def _capture_custom_functional_operator(self, namespaces_func_names=None):
        """猴子补丁 nn.functional 中的函数"""
        for module_path, func_names in namespaces_func_names.items():
            # 获取命名空间
            module_path_list = module_path.split(".")
            namespace = module_path_list[-1]
            module_path_list = module_path_list[:-1]
            module = __import__(".".join(module_path_list), fromlist=[''])
            if not hasattr(module, namespace):
                # 创建命名空间
                namespace_obj = types.ModuleType(f"{module}.{namespace}")
                setattr(module, namespace, namespace_obj)
            module = getattr(module, namespace)
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
                curr_dump_iter, iter_inner_idx = self._get_iter_idx(func_name, self.dump_node_count)
                inp_shape, inp_dtype, oup_shape, oup_dtype = self._dump_node_data(self.save_path, input, output, func_name, curr_dump_iter, iter_inner_idx)
                
                if func_name in self.attr_ops:
                    attr = dict(
                        input_lens_q = (input[4][1:]-input[4][:-1]).cpu().tolist(),
                        input_lens_kv = input[6].cpu().tolist()
                    )
                else:
                    attr = {}
                operator_info_dict = dict(
                    iter=curr_dump_iter,
                    ops_idx=iter_inner_idx,
                    module_name=func_name,     #data.module_name,
                    operator_name=func_name,
                    input_shapes=inp_shape,
                    input_dtypes=inp_dtype,
                    output_shapes=oup_shape,
                    output_dtypes=oup_dtype,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                    duration_time=end_time-start_time,
                    attr=attr,
                )
                self.append_json_line(f"{self.save_path}/operator_info.json",operator_info_dict)
                self.dump_node_count += 1
                return output
            self.patched_functions[func_name] = original_func
            setattr(module, func_name, patched_func)
            self.ops_count +=1
            
    def model_efficiency_dump(self, layer_data, model_name, dump_path="./"):
        df = pd.DataFrame(layer_data)
        group_data =df.groupby(["iter"]).agg({
            "duration_time":"sum",
            "computes_ops":"sum",
            "memory_byte":"sum",
        })
        group_data['computes_efficiency'] = group_data['computes_ops']/TCMAC/group_data['duration_time']
        group_data['memory_efficiency'] = group_data['memory_byte']/TCBW/group_data['duration_time']
        xlx_file = f"{dump_path}/{model_name.replace('/','_')}_efficiency.xlsx"
        with pd.ExcelWriter(xlx_file) as writer:
            df.to_excel(writer, sheet_name="total", index=True)
            group_data.to_excel(writer, sheet_name="agg", index=True)
            
    def dump_op_eff(self):
        if self.enable_ops_eff:
            self.model_efficiency_dump(self.operator_infos, self.model_path, dump_path=self.save_path)

TC = "TensorCore"
CC = "ShardCore"   
if torch.cuda.is_available():
    # jw2e
    Hz = 6 * 10**6 #MHz
    TCBW = 128*Hz #GB/s ->Byte/s
    TCMAC = 8192 * 2 * Hz # TFLOPS FP8
else:
    # cuda
    TCBW = 695.8 * 10**9 #GB/s ->Byte/s
    TCMAC = 37.42 * 2 * 2 * 10**12 # TFLOPS FP8


class ModelEfficienyTransformer:
    def __init__(self, model_config=None, quant_type=None,rope_width=64):
        
        self.ops_map = {
            # cc
            "VocabParallelEmbedding": self.VocabParallelEmbedding,
            "OPTLearnedPositionalEmbedding": self.VocabParallelEmbedding,
            "RMSNorm":self.RMSNorm,
            "RotaryEmbedding":self.RotaryEmbedding,
            "reshape_and_cache_flash": self.reshape_and_cache,
            "SiluAndMul": self.SiluAndMul,
            "LayerNorm": self.RMSNorm,
            "ReLU":self.ReLU,
            
            # tc
            "QKVParallelLinear":self.Linear,
            "RowParallelLinear":self.Linear,
            "ColumnParallelLinear":self.Linear,
            "MergedColumnParallelLinear":self.Linear,
            "LogitsProcessor":self.LogitsProcessor, 
            
            # attn
            "varlen_fwd": self.varlen_fwd,
        }
        self.quant_pack_size = 1
        self.quant_type=quant_type
        if self.quant_type is None and model_config:
            quantization_config = model_config.get("quantization_config", None)
            if quantization_config:
                
                bits = quantization_config.get('bits', None)
                if bits is None:
                    logger.info("not quant dtype !")
                elif bits < 8:
                    self.quant_pack_size = 8 / bits
                    self.quant_type = f"torch.int8"
                else:
                    self.quant_type = f"torch.int{quantization_config['bits']}"
                    
        
        self.rope_width=rope_width
    
    def get_ops_efficiency(self, OperatorInfo):
        if OperatorInfo['operator_name'] not in self.ops_map:
            logger.info(f" warning ops: {OperatorInfo['operator_name']} not set spec func, used Default_Ops_func")
        func = self.ops_map.get(OperatorInfo['operator_name'], self.Default_Ops_func)
        ops_type, computes_ops, memory_byte = func(OperatorInfo)
        return [ops_type, computes_ops, memory_byte, computes_ops/TCMAC/OperatorInfo['duration_time'], memory_byte/TCBW/OperatorInfo['duration_time']]
        
    def Default_Ops_func(self, OperatorInfo):
        OperatorInfo['duration_time']=1*10**(-11)
        return (CC, 0, 0)
    
    def _get_common_memory_byte(self, OperatorInfo):
        memory_byte = 0
        for i,input in enumerate(OperatorInfo['input_shapes']):
            if not input or len(input) > 3: # kv cache
                continue
            input_num = torch._utils._element_size(eval(OperatorInfo['input_dtypes'][i]))
            for _, dim in enumerate(input):
                input_num *= dim
            memory_byte += input_num
        # 输出
        for i,output in enumerate(OperatorInfo['output_shapes']):
            if not output or len(input) > 3: # kv cache
                continue
            output_num = torch._utils._element_size(eval(OperatorInfo['output_dtypes'][i]))
            for _, dim in enumerate(output): 
                output_num *= dim
            memory_byte += output_num
            
        return memory_byte
    
    def ReLU(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        return (CC, computes_ops, memory_byte,)
    
    def VocabParallelEmbedding(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        for i,output in enumerate(OperatorInfo['output_shapes']):
            if not output:
                continue
            output_num = torch._utils._element_size(eval(OperatorInfo['output_dtypes'][i]))
            for _, dim in enumerate(output): 
                output_num *= dim
            memory_byte += output_num 
        return (CC, computes_ops, memory_byte,)
    
    def RMSNorm(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        memory_byte += torch._utils._element_size(eval(OperatorInfo['input_dtypes'][0]))*OperatorInfo['input_shapes'][0][-1]
        return (CC, computes_ops, memory_byte,)
    
    def Linear(self, OperatorInfo):
        assert len(OperatorInfo['input_shapes'][0])==2
        if self.quant_type and OperatorInfo['operator_name'] != "LogitsProcessor":
            weight_byte = self.quant_type
        else:
            weight_byte = OperatorInfo['input_dtypes'][0]
        bpp = torch._utils._element_size(eval(OperatorInfo['input_dtypes'][0]))
        
        computes_ops = bpp*2*OperatorInfo['input_shapes'][0][0]*OperatorInfo['input_shapes'][0][1]*OperatorInfo['output_shapes'][0][1] 
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        w_bpp = torch._utils._element_size(eval(weight_byte)) / self.quant_pack_size
        memory_byte += w_bpp*OperatorInfo['input_shapes'][0][-1]*OperatorInfo['output_shapes'][0][-1]
        
        return (TC, computes_ops, memory_byte,)
    
    def LogitsProcessor(self, OperatorInfo):
        # assert len(OperatorInfo['input_shapes'][0])==0
        assert len(OperatorInfo['input_shapes'][0])==2
        weight_byte = OperatorInfo['input_dtypes'][0]
        bpp = torch._utils._element_size(eval(weight_byte))
        computes_ops = bpp*2*OperatorInfo['input_shapes'][0][0]*OperatorInfo['input_shapes'][0][1]*OperatorInfo['output_shapes'][0][1] 
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        
        memory_byte += bpp*OperatorInfo['input_shapes'][0][-1]*OperatorInfo['output_shapes'][0][-1]
        
        return (TC, computes_ops, memory_byte,)
    
    def RotaryEmbedding(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        memory_byte += 4*OperatorInfo['input_shapes'][0][0]*self.rope_width
        return (CC, computes_ops, memory_byte,)
    
    def reshape_and_cache(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        return (CC, computes_ops, memory_byte,)
    
    def SiluAndMul(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        return (CC, computes_ops, memory_byte,)
    
    def varlen_fwd(self, OperatorInfo):
        input_lens_q = OperatorInfo['attr']["input_lens_q"] #inputs[4][1:]-OperatorInfo.inputs[4][:-1]
        input_lens_kv = OperatorInfo['attr']["input_lens_kv"] #inputs[6]
        head_num_attn = OperatorInfo['input_shapes'][0][1]
        head_dim_q = OperatorInfo['input_shapes'][0][-1]
        head_dim_v = OperatorInfo['output_shapes'][0][-1]
        bpp =torch._utils._element_size(eval(OperatorInfo['input_dtypes'][0]))
        
        head_num_kv = OperatorInfo['input_shapes'][1][-2]
        
        # computes_ops
        computes_ops = 0
        for b, _ in enumerate(input_lens_q):
            computes_ops += bpp*input_lens_q[b] * head_num_attn * input_lens_kv[b] * (head_dim_q + head_dim_v) * 2
            
        # memory_byte
        memory_byte = 0
        for b, _ in enumerate(input_lens_q):
            memory_byte += input_lens_q[b]* head_num_attn * head_dim_q * bpp
            memory_byte += input_lens_kv[b]* head_num_kv * head_dim_q * bpp
            memory_byte += input_lens_kv[b]* head_num_kv * head_num_kv * bpp
        return (TC, computes_ops, memory_byte,)
