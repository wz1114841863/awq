import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


def make_divisible(c, divisor):
    """将一个数调整为可以被某个除数整除的最小整数."""
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    """根据输入特征维度 in_features 和分组大小 group_size,计算零值填充的宽度."""
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    """将未打包的量化权重(unpacked_qweight)打包为特定格式
        优化硬件计算效率和存储效率.
    交错存储:
        每4行权重交错排列,匹配GPU内存访问模式
    位压缩:
        4个4-bit数值打包为1个INT16
    参数:
        unpacked_qweight:未打包的量化权重,形状为 [N, K].
        interleave:交错参数,用于控制权重的交错排列.
        kstride:步长参数,用于控制权重的步长排列.
    """
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    # 将权重打包为 int16 格式.
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class ScaledActivation(nn.Module):
    """对激活值进行缩放"""

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    """实现量化线性层, 替换原始 nn.Linear"""

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0  # 输入维度需能被分组大小整除
        assert out_features % (32 // self.w_bit) == 0  # 输出维度需对齐硬件(8的倍数)
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit
        # NVIDIA GPU的Tensor Core要求矩阵宽度为8的倍数
        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",  # 存储量化后的权重, 使用 INT16 存储4-bit数据,实现4x压缩
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        # scales 和 scaled_zeros 按分组存储,额外内存占比<0.5%
        self.register_buffer(
            "scales",  # 存储缩放因子
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaled_zeros",  # 存储缩放后的零值
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",  # 存储偏置
                torch.zeros((out_features), dtype=torch.float16, device=dev),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  # shape: [768, 6]
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        # scales形状:[计算后的零值宽度*8, Out]
        # 计算后的零值宽度 = ceil(In/group_size / 8)*8
        # 确保每个CUDA线程块访问连续128-bit数据
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(  # (weight + scale_zeros) / scales
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        awq_linear.qweight = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )

        # 计算缩放后的零值(零值量化)
        zeros = zeros.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.scaled_zeros = scaled_zeros.transpose(
            1, 0
        ).contiguous()  # 将反量化公式中的乘法提前计算

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        if inputs.numel() / inputs.shape[-1] < 8:
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )  # 小批量, 向量乘法优化
        else:
            out = awq_inference_engine.gemm_forward_cuda_new(
                inputs, self.qweight, self.scales, self.scaled_zeros
            )  # - 8.0 * self.scales) 大批量, 矩阵乘法优化
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
