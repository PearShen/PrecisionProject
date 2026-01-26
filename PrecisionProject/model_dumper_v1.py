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


from PrecisionProject.utils.data import prepare_input_and_output, transfer_torch2np_data, load_model_config
from PrecisionProject.operator_capture import OperatorInfo, OperatorCaptureFramework
from PrecisionProject.model_efficiency import ModelEfficienyTransformer


class ModelDumper:
    """Dumps model structure and operator execution traces"""

    def __init__(self, framework: str = "vllm", model_path=None, 
                 enable_ops_eff: bool = True, dump_start_iter=0, dump_iter_count=3):
        """
        Initialize the model dumper.

        Args:
            framework: Framework type, either "torch" or "vllm"
            enable_enhanced_capture: Whether to enable comprehensive operator capture
        """
        self.framework = framework
        self.ignore_big_tensor_framework= ["vllm", "sglang"]
        self.dump_start_iter=dump_start_iter
        self.dump_iter_count=dump_iter_count
        self.enable_ops_eff = enable_ops_eff
        self.model_path=model_path
        
            

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

        if self.framework == "vllm":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            self._dump_vllm_model(model, input_data, output_path, model_name, iterations)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    
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
        from PrecisionProject.V1.model_dumper import ModelDumper
        self.dumper = ModelDumper(output_path, 
                    model.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model, 
                    dump_start_iter=self.dump_start_iter, dump_iter_count=self.dump_iter_count, model_path=self.model_path,
                    enable_ops_eff=self.enable_ops_eff)
        self.dumper.enable()
        outputs = self._execute_vllm_model(model, input_data)
        self.dumper.disable()
        self.dumper.dump_op_eff()

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
        except Exception as e:
            # Fallback for testing: return mock output
            raise e

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