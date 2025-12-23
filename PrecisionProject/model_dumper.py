"""
Model structure and operator execution dumper for PyTorch and vLLM frameworks.
Enhanced with comprehensive operator capture capabilities.
"""

import torch
import torch.nn as nn
import h5py
import json
import os
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd


from .utils.data import prepare_input_and_output, transfer_torch2np_data
from .operator_capture import OperatorInfo, OperatorCaptureFramework
from .model_efficiency import ModelEfficienyTransformer


class ModelDumper:
    """Dumps model structure and operator execution traces"""

    def __init__(self, framework: str = "torch", enable_enhanced_capture: bool = True, enable_ops_eff: bool = True):
        """
        Initialize the model dumper.

        Args:
            framework: Framework type, either "torch" or "vllm"
            enable_enhanced_capture: Whether to enable comprehensive operator capture
        """
        self.framework = framework
        self.ignore_big_tensor_framework= ["vllm", "sglang"]
        self.pre_operator_traces: List[OperatorInfo] = []
        self.operator_traces: List[OperatorInfo] = []
        self.iteration_count = 0
        self.pre_hooks = []
        self.hooks = []
        self.transformer_ops_head = None
        self.transformer_ops_tail = None
        self.module_name_dict = {}
        self.hook_registry = set()
        self.ops_count=0
        self.enable_enhanced_capture = enable_enhanced_capture
        self.enable_ops_eff = enable_ops_eff and framework in self.ignore_big_tensor_framework
        
        
        if self.enable_enhanced_capture:
            self.enhanced_capture_manager = OperatorCaptureFramework(operator_traces=self.operator_traces)
            self.capture_operator_info()
        if self.enable_ops_eff:
            self.ops_eff_manager = ModelEfficienyTransformer()
            
            
    def capture_operator_info(self):
        if self.enable_enhanced_capture:
            self.enhanced_capture_manager.capture_torch_functional()
            self.enhanced_capture_manager.capture_custom_operator()
            self.module_name_dict.update(self.enhanced_capture_manager.module_name_dict)
            self.ops_count += self.enhanced_capture_manager.ops_count

    def dump_model_execution(self,
                           model: Any,
                           input_data: Any,
                           output_path: str,
                           model_name: str = "model",
                           iterations: int = 1,) -> None:
        """
        Dump model execution with operator traces.

        Args:
            model: The model to dump
            input_data: Input data for the model
            output_path: Path to save the dump
            model_name: Name of the model
            iterations: Number of forward passes to record
        """

        if self.framework == "torch":
            self._dump_torch_model(model, input_data, output_path, model_name, iterations)
        elif self.framework == "vllm":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            self._dump_vllm_model(model, input_data, output_path, model_name, iterations)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _dump_torch_model(self,
                         model: nn.Module,
                         input_data: torch.Tensor,
                         output_path: str,
                         model_name: str,
                         iterations: int) -> None:
        """Dump PyTorch model execution"""
        self._register_torch_hooks(model, "")

        os.makedirs(output_path, exist_ok=True)

        # Run a forward pass to collect layer information
        initial_output = self._execute_and_collect_layer_info(model, input_data)
        
        # Save enhanced model structure
        model_info = {
            "framework": "torch",
            "model_name": model_name,
            "model_type": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_info": {
                "shape": list(input_data.shape),
                "dtype": str(input_data.dtype)
            },
            "output_info": {
                "shape": list(initial_output.shape) if hasattr(initial_output, 'shape') else [],
                "dtype": str(initial_output.dtype) if hasattr(initial_output, 'dtype') else str(type(initial_output).__name__)
            },
            "layers": self._get_enhanced_torch_model_structure(model)
        }

        with open(os.path.join(output_path, f"model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        # Run forward passes
        # iterations=1
        for i in range(iterations):
            self.iteration_count = i
            with torch.no_grad():
                _ = model(input_data)

        # Save operator traces to HDF5
        self._save_traces_hdf5(output_path, model_name)

        # Clean up hooks
        self._remove_hooks()

    def _dump_vllm_model(self,
                        model: Any,
                        input_data: Any,
                        output_path: str,
                        model_name: str,
                        iterations: int) -> None:
        """Dump vLLM model execution"""
        SamplingParams = None
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            # Create a dummy SamplingParams for use without vLLM installed
            class DummySamplingParams:
                def __init__(self, *args, **kwargs):
                    pass
            SamplingParams = DummySamplingParams

        os.makedirs(output_path, exist_ok=True)
        self._register_torch_hooks(model.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model, "")
        # self.dump_model(model.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model, model_name, output_path)
        # self._register_torch_hooks(model.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.sampler, "sampler")
        
        # Execute model to get outputs
        outputs = self._execute_vllm_model(model, input_data)

        # Save enhanced model info
        model_info = {
            "framework": "vllm",
            "model_name": model_name,
            "model_type": "vLLM",
            "parameters": "N/A",  # vLLM handles parameter counting differently
            "input_info": {
                "type": "text_or_sampling_params",
                "sample_input": str(input_data)[:200] + "..." if len(str(input_data)) > 200 else str(input_data)
            },
            "output_info": {
                "type": "generated_text",
                "sample_output": str(outputs)[:200] + "..." if len(str(outputs)) > 200 else str(outputs),
                "num_outputs": len(outputs) if isinstance(outputs, list) else 1
            },
            "layers": self._get_vllm_model_structure(model)
        }
        with open(os.path.join(output_path, f"model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        if self.enable_ops_eff:
            self.model_efficiency_dump(model_info["layers"], model_name, output_path)
        # For vLLM, we need to patch the model to capture operator traces
        # self._patch_vllm_model(model, model_name)
        self._save_traces_hdf5(output_path, model_name)
        # Clean up hooks
        
        print(f".....ops_cout: {self.ops_count}, {len(self.hooks)}, {len(self.operator_traces)}, ")
        self._remove_hooks()

    def _should_register_hook(self, module):
        """判断是否需要注册钩子"""
        module_id = id(module)
        if module_id in self.hook_registry:
            return False
        return True
    
    def pre_hook_fn(self, name, module, input):
        # Convert inputs and outputs to numpy arrays
        # inputs_np, outputs_np = prepare_input_and_output(input, output)
        inputs_np = []

        pre_operator_info = OperatorInfo(
            module=module,
            module_name=name,
            operator_name=type(module).__name__,
            input_shapes=[list(inp.shape) if hasattr(inp, 'shape') else [] for inp in input],
            input_dtypes=[str(inp.dtype) if hasattr(inp, 'dtype') else str(type(inp).__name__) for inp in input],
            inputs=inputs_np,
            start_timestamp=time.perf_counter()
        )
        self.pre_operator_traces.append(pre_operator_info)
        return input

    
    def hook_fn(self, name, module, input, output):
        # Convert inputs and outputs to numpy arrays
        # inputs_np, outputs_np = prepare_input_and_output(input, output)
        timestamp=time.perf_counter()
        inputs_np = []
        outputs_np = []
        
        for inp in input:
            if isinstance(inp, torch.Tensor):
                inputs_np.append(transfer_torch2np_data(inp))
            else:
                inputs_np.append(np.array(inp))

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
            module=module,
            module_name=name,
            operator_name=type(module).__name__,
            input_shapes=[list(inp.shape) if hasattr(inp, 'shape') else [] for inp in input],
            input_dtypes=[str(inp.dtype) if hasattr(inp, 'dtype') else str(type(inp).__name__) for inp in input],
            output_shapes=[list(out.shape) if hasattr(out, 'shape') else [] for out in (outputs_np if isinstance(output, (list, tuple)) else [output])],
            # output_dtypes=[str(out.dtype) if hasattr(out, 'dtype') else str(type(out).__name__) for out in (outputs_np if isinstance(output, (list, tuple)) else [output])],
            output_dtypes=[str(out.dtype) if hasattr(out, 'dtype') else str(type(out).__name__) for out in (output if isinstance(output, (list, tuple)) else [output])],
            inputs=inputs_np,
            outputs=outputs_np,
            start_timestamp=self.pre_operator_traces[-1].start_timestamp,
            end_timestamp=timestamp,
            duration_time=timestamp-self.pre_operator_traces[-1].start_timestamp
        )
        self.operator_traces.append(operator_info)
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
        if self.framework != "torch":
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
    
    def _patch_vllm_model(self, model: Any, model_name: str) -> None:
        """Patch vLLM model to capture operator traces"""
        # This is a simplified version - vLLM patching requires more sophisticated hooking
        # For now, we'll create a basic trace
        operator_info = OperatorInfo(
            iteration=self.iteration_count,
            model_name=model_name,
            layer_name="vllm_model",
            operator_name="generate",
            input_shapes=[[len(model_name)]],
            output_shapes=[[len(model_name)]],
            input_dtypes=["str"],
            output_dtypes=["str"],
            inputs=[np.array([model_name])],
            outputs=[np.array([model_name])],
            start_timestamp=time.time(),
            end_timestamp=time.time(),
            duration_time=time.time()
        )
        self.operator_traces.append(operator_info)

    def _execute_and_collect_layer_info(self, model: nn.Module, input_data: torch.Tensor) -> Any:
        """Execute model and collect layer information during forward pass"""
        with torch.no_grad():
            output = model(input_data)
        return output

    def _get_enhanced_torch_model_structure(self, model: nn.Module) -> List[Dict]:
        """Extract enhanced PyTorch model structure with detailed information"""
        structure = []

        for name, module in model.named_modules():
            if name:  # Skip root module
                layer_info = {
                    "layer_index": len(structure),
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable": any(p.requires_grad for p in module.parameters()),
                    "parameter_shapes": [list(p.shape) for p in module.parameters()],
                    "parameter_dtypes": [str(p.dtype) for p in module.parameters()]
                }

                # Add layer-specific information
                if isinstance(module, nn.Linear):
                    layer_info.update({
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "bias": module.bias is not None
                    })
                elif isinstance(module, nn.Conv2d):
                    layer_info.update({
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "bias": module.bias is not None
                    })
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    layer_info.update({
                        "num_features": module.num_features,
                        "eps": module.eps,
                        "momentum": module.momentum,
                        "affine": module.affine,
                        "track_running_stats": module.track_running_stats
                    })
                elif isinstance(module, nn.ReLU):
                    layer_info.update({
                        "inplace": module.inplace
                    })
                elif isinstance(module, nn.Dropout):
                    layer_info.update({
                        "p": module.p,
                        "inplace": module.inplace
                    })
                elif isinstance(module, nn.Softmax):
                    layer_info.update({
                        "dim": module.dim
                    })
                elif isinstance(module, nn.Flatten):
                    layer_info.update({
                        "start_dim": module.start_dim,
                        "end_dim": module.end_dim
                    })
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    layer_info.update({
                        "output_size": module.output_size
                    })
                elif isinstance(module, nn.Sequential):
                    layer_info.update({
                        "num_layers": len(module)
                    })

                structure.append(layer_info)

        return structure

    def _get_torch_model_structure(self, model: nn.Module) -> List[Dict]:
        """Extract PyTorch model structure (legacy method)"""
        structure = []
        for name, module in model.named_modules():
            if name:
                structure.append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                })
        return structure

    def _execute_vllm_model(self, model: Any, input_data: Any) -> Any:
        """Execute vLLM model and get outputs"""
        try:
            if isinstance(input_data, str):
                outputs = model.generate(input_data)
            else:
                # Mock SamplingParams for testing without vLLM
                class DummySamplingParams:
                    def __init__(self, *args, **kwargs):
                        pass
                SamplingParams = DummySamplingParams

                try:
                    from vllm import SamplingParams
                except ImportError:
                    pass

                sampling_params = SamplingParams(**input_data.get("params", {}))
                outputs = model.generate(input_data["prompts"], sampling_params)
            res = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt:    {prompt!r}")
                print(f"Output:    {generated_text!r}")
                print("-" * 60)
                res.append(generated_text)
            return res
        except:
            # Fallback for testing: return mock output
            return ["Generated text response from mock model"]

    def _get_vllm_model_structure(self, model: Any) -> List[Dict]:
        """Extract enhanced vLLM model structure"""
        # import pdb
        # pdb.set_trace()
        structure = []
        module_static = {}
        iter_cout = -1
        iter_ops_idx = 0
        parent_module_name = self.transformer_ops_tail
        for i, data in enumerate(self.operator_traces):
            if (self.framework != "torch" and data.module_name == self.transformer_ops_head and parent_module_name == self.transformer_ops_tail) or \
                (self.framework == "torch" and i%self.ops_count == 0):
                module_static = {}
                iter_cout += 1
                iter_ops_idx = 0
            module_ops_count = module_static.get(data.module, 0)
            module_static[data.module] = module_ops_count+1
            dup_module_lens = len(self.module_name_dict[data.module])
            obj = dict(
                iter=iter_cout,
                ops_idx=iter_ops_idx,
                module_name=self.module_name_dict[data.module][module_ops_count%dup_module_lens],     #data.module_name,
                operator_name=data.operator_name,
                input_shapes=data.input_shapes,
                input_dtypes=data.input_dtypes,
                output_shapes=data.output_shapes,
                output_dtypes=data.output_dtypes,
                start_timestamp=data.start_timestamp,
                end_timestamp=data.end_timestamp,
                duration_time=data.duration_time,
            )
            if self.enable_ops_eff:
                ops_type, computes_ops, memory_byte, computes_efficiency, memory_efficiency = self.ops_eff_manager.get_ops_efficiency(data)
                obj["duration_time"] = data.duration_time
                obj["ops_type"] = ops_type
                obj["computes_ops"] = float(computes_ops)
                obj["memory_byte"] = float(memory_byte)
                obj["computes_efficiency"] = computes_efficiency
                obj["memory_efficiency"] = memory_efficiency
            iter_ops_idx += 1
            structure.append(obj)
            parent_module_name = obj["module_name"]
        return structure

    # Legacy method for compatibility
    def load_traces(self, hdf5_path: str) -> List[OperatorInfo]:
        """Load operator traces from HDF5 file (legacy method)"""
        traces = []

        with h5py.File(hdf5_path, "r") as f:
            for key in f.keys():
                if key.startswith("trace_"):
                    group = f[key]

                    #Extract trace data
                    trace_data = {
                        "iteration": group.attrs["iteration"],
                        "model_name": group.attrs["module_name"],
                        "layer_name": group.attrs["layer_name"],
                        "operator_name": group.attrs["operator_name"],
                        "input_shapes": json.loads(group.attrs["input_shapes"]),
                        "output_shapes": json.loads(group.attrs["output_shapes"]),
                        "input_dtypes": json.loads(group.attrs["input_dtypes"]),
                        "output_dtypes": json.loads(group.attrs["output_dtypes"]),
                        "inputs": [],
                        "outputs": [],
                    }

                    # Load input arrays
                    input_keys = [k for k in group.keys() if k.startswith("input_")]
                    for k in sorted(input_keys):
                        trace_data["inputs"].append(group[k][:])

                    # Load output arrays
                    output_keys = [k for k in group.keys() if k.startswith("output_")]
                    for k in sorted(output_keys):
                        trace_data["outputs"].append(group[k][:])

                    traces.append(trace_data)

        return traces

    def _save_traces_hdf5(self, output_path: str, model_name) -> None:
        """Save operator traces to HDF5 file"""
        hdf5_path = os.path.join(output_path, f"operator_traces.h5")

        with h5py.File(hdf5_path, "w") as f:
            iter_cout = -1
            parent_module_name = self.transformer_ops_tail
            module_static = {}
            iter_ops_idx = 0
            for i, trace in enumerate(self.operator_traces):
                if (self.framework != "torch" and trace.module_name == self.transformer_ops_head and parent_module_name == self.transformer_ops_tail) \
                    or (self.framework == "torch" and i%self.ops_count == 0):
                    module_static = {}
                    iter_cout += 1
                    iter_ops_idx = 0
                module_ops_count = module_static.get(trace.module, 0)
                module_static[trace.module] = module_ops_count+1
                dup_module_lens = len(self.module_name_dict[trace.module])
                iter=iter_cout,
                group = f.create_group(f"trace_{i:06d}")
                group.attrs["iteration"] = iter
                group.attrs["ops_idx"] = iter_ops_idx
                # Save metadata
                group.attrs["module_name"] = self.module_name_dict[trace.module][module_ops_count%dup_module_lens]
                group.attrs["operator_name"] = trace.operator_name

                # Save shapes and dtypes
                group.attrs["input_shapes"] = json.dumps(trace.input_shapes)
                group.attrs["output_shapes"] = json.dumps(trace.output_shapes)
                group.attrs["input_dtypes"] = json.dumps(trace.input_dtypes)
                group.attrs["output_dtypes"] = json.dumps(trace.output_dtypes)
                
                iter_ops_idx += 1
                parent_module_name = group.attrs["module_name"]

                # Save inputs
                for j, inp in enumerate(trace.inputs):
                    if isinstance(inp, np.ndarray):
                        # Check if it's a string array
                        if inp.dtype.kind in ['U', 'S', 'O']:
                            # Encode strings as bytes for HDF5
                            if inp.dtype.kind == 'U':
                                if inp.ndim > 0:
                                    encoded = np.array([s.encode('utf-8') for s in inp])
                                else:
                                    encoded = np.array([inp.item().encode('utf-8')])
                                group.create_dataset(f"input_{j}", data=encoded)
                            else:
                                group.create_dataset(f"input_{j}", data=inp.astype('S'))
                        else:
                            if self.framework in self.ignore_big_tensor_framework and inp.ndim > 3:
                                inp = np.array([])
                            group.create_dataset(f"input_{j}", data=inp)
                    else:
                        # Non-array inputs
                        group.create_dataset(f"input_{j}", data=np.array([str(inp)], dtype='S'))

                # Save outputs
                for j, out in enumerate(trace.outputs):
                    if isinstance(out, np.ndarray):
                        # Check if it's a string array
                        if out.dtype.kind in ['U', 'S', 'O']:
                            # Encode strings as bytes for HDF5
                            if out.dtype.kind == 'U':
                                encoded = np.array([s.encode('utf-8') for s in out])
                                group.create_dataset(f"output_{j}", data=encoded)
                            else:
                                group.create_dataset(f"output_{j}", data=out.astype('S'))
                        else:
                            group.create_dataset(f"output_{j}", data=out)
                    else:
                        # Non-array outputs
                        group.create_dataset(f"output_{j}", data=np.array([str(out)], dtype='S'))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hooks_name = {}
        # Clean up hooks
        self.hook_registry.clear()
        # Clean up hooks
        self.operator_traces.clear()
        self.pre_operator_traces.clear()
        self.pre_hooks.clear()
        
        self.ops_count=0
        self.module_name_dict = {}
        self.transformer_ops_head = None
        self.transformer_ops_tail = None
        if self.enable_enhanced_capture:
            self.enhanced_capture_manager.restore_functional()
    
    def model_efficiency_dump(self, layer_data, model_name, dump_path="./"):
        df = pd.DataFrame(layer_data)
        group_data =df.groupby(["iter"]).agg({
            "duration_time":"sum",
            "computes_ops":"sum",
            "memory_byte":"sum",
            # "computes_efficiency":'mean',
            # "memory_efficiency":'mean',
        })
        group_data['computes_efficiency'] = group_data['computes_ops']/self.ops_eff_manager.tcMac/group_data['duration_time']
        group_data['memory_efficiency'] = group_data['memory_byte']/self.ops_eff_manager.tcBW/group_data['duration_time']
        xlx_file = f"{dump_path}/{model_name.replace('/','_')}_efficiency.xlsx"
        with pd.ExcelWriter(xlx_file) as writer:
            df.to_excel(writer, sheet_name="total", index=True)
            group_data.to_excel(writer, sheet_name="agg", index=True)
        
    
    def dump_model(self, model, model_name, dump_path="./"):
        # 动态轴处理
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'token_type_ids': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'}
        }

        torch.onnx.export(
            model,
            (torch.randint(0, 100, (1, 128)), torch.ones((1, 128)), torch.zeros((1, 128))),
            f"{dump_path}/{model_name}.onnx",
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['output'],
            dynamic_axes=None,#dynamic_axes,
            opset_version=14,
            export_params=False
        )