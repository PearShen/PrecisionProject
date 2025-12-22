import torch
import logging

logger = logging.getLogger(__name__)


TC = "TensorCore"
CC = "ShardCore"

class ModelEfficienyTransformer:
    def __init__(self,quant_type=None,rope_width=64):
        
        self.ops_map = {
            # cc
            "VocabParallelEmbedding": self.VocabParallelEmbedding,
            "RMSNorm":self.RMSNorm,
            "RotaryEmbedding":self.RotaryEmbedding,
            "reshape_and_cache_flash": self.reshape_and_cache,
            "SiluAndMul": self.SiluAndMul,
            
            # tc
            "QKVParallelLinear":self.Linear,
            "RowParallelLinear":self.Linear,
            "MergedColumnParallelLinear":self.Linear,
            "LogitsProcessor":self.LogitsProcessor, 
            
            # attn
            "varlen_fwd": self.varlen_fwd,
        }
        self.quant_type=None
        self.rope_width=rope_width

        self.tcBW = 695.8 * 10**9 #GB/s ->Byte/s
        self.tcMac = 37.42 * 2 * 2 * 10**12 # TFLOPS FP8
    
    def get_ops_efficiency(self, OperatorInfo):
        if OperatorInfo.operator_name not in self.ops_map:
            logger.info(f" warning ops: {OperatorInfo.operator_name} not set spec func, used Default_Ops_func")
        func = self.ops_map.get(OperatorInfo.operator_name, self.Default_Ops_func)
        ops_type, computes_ops, memory_byte = func(OperatorInfo)
        return [ops_type, computes_ops, memory_byte, computes_ops/self.tcMac/OperatorInfo.timestamp, memory_byte/self.tcBW/OperatorInfo.timestamp]
        
    def Default_Ops_func(self, OperatorInfo):
        OperatorInfo.timestamp=1*10**(-11)
        return (CC, 0, 0)
    
    def _get_common_memory_byte(self, OperatorInfo):
        memory_byte = 0
        for i,input in enumerate(OperatorInfo.input_shapes):
            if not input or len(input) > 3: # kv cache
                continue
            input_num = torch._utils._element_size(eval(OperatorInfo.input_dtypes[i]))
            for _, dim in enumerate(input):
                input_num *= dim
            memory_byte += input_num
        # 输出
        for i,output in enumerate(OperatorInfo.output_shapes):
            if not output or len(input) > 3: # kv cache
                continue
            output_num = torch._utils._element_size(eval(OperatorInfo.output_dtypes[i]))
            for _, dim in enumerate(output): 
                output_num *= dim
            memory_byte += output_num
            
        return memory_byte
    
    def VocabParallelEmbedding(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        for i,output in enumerate(OperatorInfo.output_shapes):
            if not output:
                continue
            output_num = torch._utils._element_size(eval(OperatorInfo.output_dtypes[i]))
            for _, dim in enumerate(output): 
                output_num *= dim
            memory_byte += output_num 
        return (CC, computes_ops, memory_byte,)
    
    def RMSNorm(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        memory_byte += torch._utils._element_size(eval(OperatorInfo.input_dtypes[0]))*OperatorInfo.input_shapes[0][-1]
        return (CC, computes_ops, memory_byte,)
    
    def Linear(self, OperatorInfo):
        assert len(OperatorInfo.input_shapes[0])==2
        if self.quant_type and OperatorInfo.operator_name != "LogitsProcessor":
            weight_byte = self.quant_type
        else:
            weight_byte = OperatorInfo.input_dtypes[0]
        bpp = torch._utils._element_size(eval(weight_byte))
        
        computes_ops = bpp*2*OperatorInfo.input_shapes[0][0]*OperatorInfo.input_shapes[0][1]*OperatorInfo.output_shapes[0][1] 
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        
        memory_byte += bpp*OperatorInfo.input_shapes[0][-1]*OperatorInfo.output_shapes[0][-1]
        
        return (TC, computes_ops, memory_byte,)
    
    def LogitsProcessor(self, OperatorInfo):
        assert len(OperatorInfo.input_shapes[0])==0
        assert len(OperatorInfo.input_shapes[1])==2
        weight_byte = OperatorInfo.input_dtypes[1]
        bpp = torch._utils._element_size(eval(weight_byte))
        computes_ops = bpp*2*OperatorInfo.input_shapes[1][0]*OperatorInfo.input_shapes[1][1]*OperatorInfo.output_shapes[0][1] 
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add weight size
        
        memory_byte += bpp*OperatorInfo.input_shapes[1][-1]*OperatorInfo.output_shapes[0][-1]
        
        return (TC, computes_ops, memory_byte,)
    
    def RotaryEmbedding(self, OperatorInfo):
        computes_ops = 0
        memory_byte = self._get_common_memory_byte(OperatorInfo)
        # add scale
        memory_byte += 4*OperatorInfo.input_shapes[0][0]*self.rope_width
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
        input_lens_q = OperatorInfo.inputs[4][1:]-OperatorInfo.inputs[4][:-1]
        input_lens_kv = OperatorInfo.inputs[6]
        head_num_attn = OperatorInfo.input_shapes[0][1]
        head_dim_q = OperatorInfo.input_shapes[0][-1]
        head_dim_v = OperatorInfo.output_shapes[0][-1]
        bpp =torch._utils._element_size(eval(OperatorInfo.input_dtypes[0]))
        
        head_num_kv = OperatorInfo.input_shapes[1][-2]
        
        # computes_ops
        computes_ops = 0
        for b, _ in enumerate(input_lens_q):
            computes_ops += bpp*input_lens_q[b]* head_num_attn * input_lens_kv[b] * (head_dim_q + head_dim_v) * 2
            
        # memory_byte
        memory_byte = 0
        for b, _ in enumerate(input_lens_q):
            memory_byte += input_lens_q[b]* head_num_attn * head_dim_q * bpp
            memory_byte += input_lens_kv[b]* head_num_kv * head_dim_q * bpp
            memory_byte += input_lens_kv[b]* head_num_kv * head_num_kv * bpp
        return (TC, computes_ops, memory_byte,)
            
            
            