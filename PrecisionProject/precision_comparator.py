"""
Precision comparison utilities for comparing model outputs.
"""

import numpy as np
import h5py
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import torch
import logging

logger = logging.getLogger(__name__)


torch.set_printoptions(precision=8)

@dataclass
class ComparisonResult:
    """Result of precision comparison"""
    operator_name: str
    module_name: str
    iteration: int
    ops_idx:int
    absolute_error_input: float
    relative_error_input: float
    cosine_similarity_input: float
    absolute_error_output: float
    relative_error_output: float
    cosine_similarity_output: float
    absolute_error: float
    relative_error: float
    cosine_similarity: float
    passed: bool
    details: Dict[str, Any]


@dataclass
class ComparisonConfig:
    """Configuration for precision comparison"""
    abs_error_threshold: float = 1e-5
    rel_error_threshold: float = 1e-5
    cosine_similarity_threshold: float = 0.99999
    compare_inputs: bool = True
    compare_outputs: bool = True
    verbose: bool = True


class PrecisionComparator:
    """Compare precision between golden and test outputs"""

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """
        Initialize the precision comparator.

        Args:
            config: Configuration for comparison thresholds
        """
        self.config = config or ComparisonConfig()

    def compare_traces(self,
                      golden_path: str,
                      test_path: str,
                      output_path: Optional[str] = None) -> Tuple[List[ComparisonResult], Dict[str, float]]:
        """
        Compare golden and test traces.

        Args:
            golden_path: Path to golden reference data
            test_path: Path to test data
            output_path: Optional path to save comparison results

        Returns:
            Tuple of (comparison results, summary statistics)
        """
        # Load traces
        golden_traces = self._load_traces(golden_path)
        test_traces = self._load_traces(test_path)
        results = []
        if len(golden_traces) != len(test_traces):
            logger.warning(f"Number of traces mismatch: golden={len(golden_traces)}, test={len(test_traces)}")
            # 缺少部分 ops varlen_fwd
            reduntant_ops = 0
            for i, golden in enumerate(golden_traces):
                if golden["operator_name"] == "varlen_fwd":
                    reduntant_ops += 1
                    continue
                test = test_traces[i-reduntant_ops]
                result = self._compare_single_trace(golden, test)
                results.append(result)
        else:
            for i, (golden, test) in enumerate(zip(golden_traces, test_traces)):
                if golden["operator_name"] == "varlen_fwd":
                    continue
                result = self._compare_single_trace(golden, test)
                results.append(result)
        # Calculate summary statistics
        summary = self._calculate_summary(results)

        # Save results if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            self._save_results(results, summary, output_path)

        return results, summary

    def compare_tensors(self,
                       golden_tensor: np.ndarray,
                       test_tensor: np.ndarray, torch_dtype: torch.dtype) -> [float, float, float]:
        """
        Compare two tensors and compute error metrics.

        Args:
            golden_tensor: Reference tensor
            test_tensor: Test tensor

        Returns:
            Tuple of (absolute_error, relative_error, cosine_similarity)
        """
        if golden_tensor.shape != test_tensor.shape:
            raise ValueError(f"Shape mismatch: {golden_tensor.shape} vs {test_tensor.shape}")

        # Convert to tensors for computation
        if not isinstance(golden_tensor, np.ndarray):
            golden_tensor = np.array(golden_tensor)
        if not isinstance(test_tensor, np.ndarray):
            test_tensor = np.array(test_tensor)

        # Flatten for comparison
        golden_flat = torch.from_numpy(golden_tensor).view(torch_dtype).flatten()#.astype(np.float64)
        test_flat = torch.from_numpy(test_tensor).view(torch_dtype).flatten()#.astype(np.float64)

        diff = torch.abs(golden_flat-test_flat)
        # Absolute error
        abs_error = torch.max(diff)

        # Relative error
        rel_error = torch.max(diff / (golden_flat + 1e-12))

        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(test_flat.to(torch.float64).view(1,-1),golden_flat.to(torch.float64).view(1,-1))
        return abs_error.item(), rel_error.item(), cosine_sim.item()

    def _load_traces(self, path: str) -> List[Dict]:
        """Load traces from HDF5 file"""
        traces = []
        hdf5_path = os.path.join(path, "operator_traces.h5")

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Traces file not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f.keys()):
                if key.startswith("trace_"):
                    group = f[key]
                    trace = self._extract_trace_from_group(group)
                    traces.append(trace)

        return traces

    def _extract_trace_from_group(self, group: h5py.Group) -> Dict:
        """Extract trace data from HDF5 group"""
        trace = {
            "iteration": group.attrs["iteration"],
            "ops_idx": group.attrs["ops_idx"],
            "module_name": group.attrs["module_name"],
            "operator_name": group.attrs["operator_name"],
            "input_shapes": json.loads(group.attrs["input_shapes"]),
            "output_shapes": json.loads(group.attrs["output_shapes"]),
            "input_dtypes": json.loads(group.attrs["input_dtypes"]),
            "output_dtypes": json.loads(group.attrs["output_dtypes"]),
            "inputs": [],
            "outputs": []
        }

        # Load inputs
        input_keys = [k for k in group.keys() if k.startswith("input_")]
        for idx, k in enumerate(input_keys):
            if group[k].shape :
                trace["inputs"].append(group[k][:])
            else:
                trace["inputs"].append(np.array([]))

        # Load outputs
        output_keys = [k for k in group.keys() if k.startswith("output_")]
        for idx, k in enumerate(output_keys):
            if group[k].shape:
                trace["outputs"].append(group[k][:])
            else:
                trace["outputs"].append(np.array([]))

        return trace

    def _compare_single_trace(self, golden: Dict, test: Dict) -> ComparisonResult:
        """Compare a single trace between golden and test"""
        # Basic validation
        # if golden["iteration"] != test["iteration"]:
        #     raise ValueError(f"Iteration mismatch: {golden['iteration']} vs {test['iteration']}")
        # if golden["ops_idx"] != test["ops_idx"]:
        #     raise ValueError(f"ops_idx mismatch: {golden['ops_idx']} vs {test['ops_idx']}")
        # if golden["module_name"] != test["module_name"]:
        #     raise ValueError(f"Layer name mismatch: {golden['module_name']} vs {test['module_name']}")
        # if golden["operator_name"] != test["operator_name"]:
        #     raise ValueError(f"Operator name mismatch: {golden['operator_name']} vs {test['operator_name']}")

        # Compare inputs if configured
        input_errors = []
        if self.config.compare_inputs and len(golden["inputs"]) == len(test["inputs"]):
            for i, (golden_input, test_input) in enumerate(zip(golden["inputs"], test["inputs"])):
                # skip empty tennsor
                if not golden_input.shape or golden_input.shape == (0,):
                    continue
                # skip non digital tensor
                if golden_input.dtype.kind in ['U', 'S', 'O']:
                    continue
                abs_err, rel_err, cos_sim = self.compare_tensors(golden_input, test_input, eval(golden["input_dtypes"][i]))
                input_errors.append((abs_err, rel_err, cos_sim))

        # Compare outputs if configured
        output_errors = []
        if self.config.compare_outputs and len(golden["outputs"]) == len(test["outputs"]):
            for i, (golden_output, test_output) in enumerate(zip(golden["outputs"], test["outputs"])):
                # skip empty tennsor
                if not golden_output.shape or golden_output.shape == (0,):
                    continue
                # skip non digital tensor
                if golden_output.dtype.kind in ['U', 'S', 'O']:
                    continue
                abs_err, rel_err, cos_sim = self.compare_tensors(golden_output, test_output, eval(golden["output_dtypes"][i]))
                output_errors.append((abs_err, rel_err, cos_sim))

        
        # Calculate input errors
        # if input_errors:
        #     abs_error_input = np.mean([err[0] for err in input_errors])
        #     rel_error_input = np.mean([err[1] for err in input_errors])
        #     cosine_similarity_input = np.mean([err[2] for err in input_errors])
        # else:
        abs_error_input = rel_error_input = 0.0
        cosine_similarity_input = 1.0
            
        # Calculate output errors
        if output_errors:
            abs_error_output = np.mean([err[0] for err in output_errors])
            rel_error_output = np.mean([err[1] for err in output_errors])
            cosine_similarity_output = np.mean([err[2] for err in output_errors])
        else:
            abs_error_output = rel_error_output = 0.0
            cosine_similarity_output = 1.0
        
        # Calculate overall errors
        all_errors = output_errors
        if all_errors:
            abs_error = np.mean([err[0] for err in all_errors])
            rel_error = np.mean([err[1] for err in all_errors])
            cosine_similarity = np.mean([err[2] for err in all_errors])
        else:
            abs_error = rel_error = 0.0
            cosine_similarity = 1.0

        # Determine if test passed
        passed = (
            # abs_error <= self.config.abs_error_threshold and
            # rel_error <= self.config.rel_error_threshold and
            cosine_similarity >= self.config.cosine_similarity_threshold
        )

        return ComparisonResult(
            operator_name=golden["operator_name"],
            module_name=golden["module_name"],
            iteration=golden["iteration"],
            ops_idx=golden["ops_idx"],
            absolute_error_input=abs_error_input,
            relative_error_input=rel_error_input,
            cosine_similarity_input=cosine_similarity_input,
            
            absolute_error_output=abs_error_output,
            relative_error_output=rel_error_output,
            cosine_similarity_output=cosine_similarity_output,
            
            absolute_error=abs_error,
            relative_error=rel_error,
            cosine_similarity=cosine_similarity,
            
            passed=passed,
            details={
                "input_count": len(golden["inputs"]),
                "output_count": len(golden["outputs"]),
                "input_errors": input_errors,
                "output_errors": output_errors
            }
        )

    def _calculate_summary(self, results: List[ComparisonResult]) -> Dict[str, float]:
        """Calculate summary statistics"""
        if not results:
            return {}

        abs_errors = [r.absolute_error for r in results]
        rel_errors = [r.relative_error for r in results]
        cosine_sims = [r.cosine_similarity for r in results]
        passed_count = sum(1 for r in results if r.passed)
        failed_ops = [dict(operator_name=r.operator_name,
                           iteration=int(r.iteration),
                           ops_idx=int(r.ops_idx)) for r in results if not r.passed]

        return {
            "total_comparisons": int(len(results)),
            "passed_comparisons": int(passed_count),
            "failed_comparisons": int(len(results) - passed_count),
            "failed_ops": failed_ops,
            "pass_rate": passed_count / len(results),
            "max_absolute_error": np.max(abs_errors),
            "mean_absolute_error": np.mean(abs_errors),
            "max_relative_error": np.max(rel_errors),
            "mean_relative_error": np.mean(rel_errors),
            "min_cosine_similarity": np.min(cosine_sims),
            "mean_cosine_similarity": np.mean(cosine_sims)
        }

    def _save_results(self, results: List[ComparisonResult], summary: Dict, output_path: str):
        """Save comparison results to files"""
        # Save detailed results as JSON
        results_data = []
        for r in results:
            results_data.append({
                "operator_name": r.operator_name,
                "module_name": r.module_name,
                "iteration": int(r.iteration),
                
                "absolute_error_input": float(r.absolute_error_input),
                "relative_error_input": float(r.relative_error_input),
                "cosine_similarity_input": float(r.cosine_similarity_input),
                
                "absolute_error_output": float(r.absolute_error_output),
                "relative_error_output": float(r.relative_error_output),
                "cosine_similarity_output": float(r.cosine_similarity_output),
                
                "absolute_error": float(r.absolute_error),
                "relative_error": float(r.relative_error),
                "cosine_similarity": float(r.cosine_similarity),
                "passed": bool(r.passed),
                "details": r.details
            })

        with open(os.path.join(output_path, "comparison_results.json"), "w") as f:
            json.dump({"results": results_data, "summary": summary}, f, indent=2)

        # Save summary as text
        with open(os.path.join(output_path, "summary.txt"), "w") as f:
            f.write("Precision Comparison Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total comparisons: {summary['total_comparisons']}\n")
            f.write(f"Passed: {summary['passed_comparisons']}\n")
            f.write(f"Failed: {summary['failed_comparisons']}\n")
            f.write(f"Pass rate: {summary['pass_rate']:.2%}\n")
            f.write(f"\nError Statistics:\n")
            f.write(f"Max absolute error: {summary['max_absolute_error']:.2e}\n")
            f.write(f"Mean absolute error: {summary['mean_absolute_error']:.2e}\n")
            f.write(f"Max relative error: {summary['max_relative_error']:.2e}\n")
            f.write(f"Mean relative error: {summary['mean_relative_error']:.2e}\n")
            f.write(f"Min cosine similarity: {summary['min_cosine_similarity']:.6f}\n")
            f.write(f"Mean cosine similarity: {summary['mean_cosine_similarity']:.6f}\n")

        if self.config.verbose:
            print(f"Results saved to {output_path}")
            print(f"Pass rate: {summary['pass_rate']:.2%}")
            print(f"Mean absolute error: {summary['mean_absolute_error']:.2e}")
            print(f"Mean relative error: {summary['mean_relative_error']:.2e}")