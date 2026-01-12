"""
Main entry point for PrecisionProject
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path



def create_vllm_model_and_input(model_name="/home/shenpeng/workspace/models/Qwen3-0.6B-GPTQ-Int4"):
    """Create a simple test model"""
    from vllm import LLM, SamplingParams 
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["DUMP_DATA"]= "1"
    os.environ["DUMP_DATA_PATH"]= "/home/shenpeng/workspace/PrecisionProject/data/vllm/"
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # model_name = "facebook/opt-125m"
    # model_name = "nickypro/tinyllama-42M-fp32"
    # model_name = "/home/shenpeng/workspace/models/Qwen3-8B-GPTQ-Int4"
    llm = LLM(model=model_name, enforce_eager=True, gpu_memory_utilization=0.4)
    
    params = {"temperature": 0.0, "max_tokens": 2}
    sampling_params = SamplingParams(**params)
    outputs = llm.generate(prompts, sampling_params)
    res = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
        res.append(generated_text)
        
def compare_result(golden_path=None, test_path=None, output_path=None, abs_threshold=0.001, rel_threshold=0.001, cosine_threshold=0.999999):
    from PrecisionProject.V1.precision_comparator import PrecisionComparator, ComparisonConfig
    golden_path = "/home/shenpeng/workspace/PrecisionProject/data/vllm_golden/"
    test_path = "/home/shenpeng/workspace/PrecisionProject/data/vllm/"
    output_path = "/home/shenpeng/workspace/PrecisionProject/data/vllm_output/"
    config = ComparisonConfig(
        abs_error_threshold=abs_threshold,
        rel_error_threshold=rel_threshold,
        cosine_similarity_threshold=cosine_threshold
    )
    comparator = PrecisionComparator(config)
    results, summary = comparator.compare_traces(
        golden_path, test_path, output_path
    )

if __name__ == "__main__":
    create_vllm_model_and_input()
    # compare_result()