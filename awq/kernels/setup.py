from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

"""
C++ 编译器选项 (cxx):
    -g: 启用调试信息生成,方便调试。
    -O3: 优化等级,O3 是最高的优化等级,会激进优化以提高代码运行速度。
    -fopenmp 和 -lgomp: 启用 OpenMP,用于多线程并行计算。
    -std=c++17: 指定使用 C++17 标准。
    -DENABLE_BF16: 定义了一个宏 ENABLE_BF16,用于启用 BFloat16 支持。

CUDA 编译器选项 (nvcc):
    -O3: 优化代码。
    -std=c++17: 使用 C++17 标准。
    -DENABLE_BF16: 启用 BFloat16 支持。
    -U__CUDA_NO_HALF_OPERATORS__ 和类似选项:
    取消对半精度浮点(FP16)和 BFloat16 操作符及转换的限制。
    --expt-relaxed-constexpr 和 --expt-extended-lambda:
    启用 CUDA 的扩展功能,例如 lambda 表达式和常量表达式支持。
    --use_fast_math:
    启用快速数学运算,可能会降低精度,但提高性能。
    --threads=8:
    指定最大线程数为 8。

"""
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ],
}

setup(
    name="awq_inference_engine",  # 包名, 可以通过import awq_inference_engine导入
    packages=find_packages(),     # 自动查找项目中所有的 Python 子包,生成需要安装的包列表。
    ext_modules=[
        CUDAExtension(
            name="awq_inference_engine",
            sources=[
                "csrc/pybind.cpp",
                "csrc/quantization/gemm_cuda_gen.cu",
                "csrc/quantization/gemv_cuda.cu",
                "csrc/quantization_new/gemv/gemv_cuda.cu",
                "csrc/quantization_new/gemm/gemm_cuda.cu",
                "csrc/layernorm/layernorm.cu",
                "csrc/position_embedding/pos_encoding_kernels.cu",
                "csrc/attention/ft_attention.cpp",
                "csrc/attention/decoder_masked_multihead_attention.cu",
                "csrc/rope_new/fused_rope_with_pos.cu",
                "csrc/w8a8/w8a8_gemm_cuda.cu",
                "csrc/w8a8/quantization.cu",
                # "csrc/fused_layernorm/layernorm_kernels.cu"
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},  # 指定使用 BuildExtension 来处理扩展模块的构建.
    install_requires=["torch"],  # 指定该包的依赖项,即安装此包时需要安装 PyTorch
)
