"""Build private group-32 E4M3 Marlin with the same FP32 MMA accumulator as NVFP4.

The locally installed SM75 Marlin template defaults to FP16 accumulation for
this FP8 specialization. Only our private copied kernel header is modified.
"""
import os
import sys
from pathlib import Path

import torch
import tempfile
from subprocess import run

ROOT=Path(sys.argv[1])
SOURCE=Path(sys.argv[2]) / "csrc"
MOE=SOURCE/"libtorch_stable/moe/marlin_moe_wna16"
TORCH=Path(torch.__file__).parent
CUDA=Path(os.environ["CUDA_HOME"])
with tempfile.TemporaryDirectory(prefix="inkling-mx32-f32-") as tmp:
    build=Path(tmp)
    template=(MOE/"marlin_template.h").read_text()
    start=template.index("  constexpr bool use_fp16_accum =\n")
    end=template.index(";",start)+1
    (build/"marlin_template.h").write_text(template[:start]+"  constexpr bool use_fp16_accum = false;"+template[end:])
    kernels=(MOE/"sm75_kernel_float16_fe4m3fn_float16.cu").read_text()
    choices=[line.replace(", 2, 8, false>",", 2, 2, false>") for line in kernels.splitlines()
             if line.startswith("template ") and ", 2, 8, false>" in line]
    assert len(choices)==15
    (build/"kernels.cu").write_text(kernels.split("template __global__")[0]+"\n".join(choices)+"\n}\n")
    selector=(MOE/"kernel_selector.h").read_text().splitlines()
    enabled=[]
    for i,line in enumerate(selector):
        if all(x in line for x in ("b_type == vllm::kFE4M3fn","a_type == vllm::kFloat16",
                                   "s_type == vllm::kFloat16","stages == 2","group_blocks == 8")):
            enabled.extend([line.replace("group_blocks == 8","group_blocks == 2"),
                            selector[i+1].replace(", 2, 8, false>", ", 2, 2, false>")])
    assert len(enabled)==2*len(choices)
    enabled[0]=enabled[0].removeprefix("else ")
    (build/"kernel_selector.h").write_text("\n".join(enabled)+"\n")
    ops=(MOE/"ops.cu").read_text().replace("moe_wna16_marlin_gemm","mx32_gemm").replace(
        "STABLE_TORCH_LIBRARY_IMPL(_moe_C,","STABLE_TORCH_LIBRARY_IMPL(inkling_sm75_f32,")
    schema=(MOE.parent/"torch_bindings.cpp").read_text().split('"moe_wna16_marlin_gemm(')[1].split(");",1)[0]
    ops+='\nSTABLE_TORCH_LIBRARY_FRAGMENT(inkling_sm75_f32, m) {\n m.def("mx32_gemm('+schema+');\n}\n'
    (build/"ops.cu").write_text(ops)
    cargs=[str(CUDA/"bin/nvcc"),"--static-global-template-stub=false","-std=c++17","-O3","--expt-relaxed-constexpr",
           "-gencode","arch=compute_75,code=sm_75","-DUSE_CUDA","-DMARLIN_NAMESPACE_NAME=inkling_marlin_mx32f32",
           "-Xcompiler","-fPIC","-I",str(MOE),"-I",str(SOURCE),"-I",str(TORCH/"include"),"-I",str(CUDA/"include/cccl")]
    for filename in ("kernels","ops"):
        run([*cargs,"-c",str(build/(filename+".cu")),"-o",str(build/(filename+".o"))],check=True)
    run([os.environ.get("CXX", "g++"),"-shared",str(build/"kernels.o"),str(build/"ops.o"),"-L",str(TORCH/"lib"),
         "-Wl,-rpath,"+str(TORCH/"lib"),"-ltorch_cpu","-ltorch_cuda","-lc10","-L",str(CUDA/"lib64"),
         "-lcudart","-o",str(ROOT/"inkling_marlin_mx32f32.so")],check=True)
