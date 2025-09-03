import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_clip_block"]


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    """对单个线性层的权重进行自动裁剪
    通过迭代搜索最佳的最大值(max_val), 使得量化后的权重与原始权重的输出误差最小.
    参数:
        w: 模型权重.
        input_feat: 输入特征, [co, ci](输出通道数 x 输入通道数).
        n_bit: 量化位数, [n_token, ci](token 数 x 输入通道数).
        q_config: 量化配置.
        n_grid: 搜索网格的数量.
        max_shrink: 最大裁剪比例, 控制权重的裁剪范围.
        n_sample_token:采样的 token 数量,用于减少计算量.
    """
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    # 对输入特征进行采样,减少计算量.
    # 采样后的形状为 [1, n_sample_token, n_group, group_size].
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        # 分批处理权重,防止显存溢出.
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        # 计算原始权重与输入特征的乘积,并对最后一个维度求和,得到原始输出
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)  # 线性降低阈值
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)  # 截断
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)  # 伪量化
            cur_out = (input_feat * q_w).sum(dim=-1)  # 计算量化误差

            # co, 1, n_group, 1
            # 把输入激活当成"重要性权重":
            # 如果某个通道的激活值很小,即使量化误差大,对最终输出的影响也小;
            # 反之激活大的位置,误差会被放大.
            # 因此 MSE 直接在"激活×权重"的输出空间 算,而不是在权重空间算.
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):
    """对给定模块中的线性层进行自动裁剪
        权重里常有 1% 左右的"离群大值",直接按全域 [-w_max, w_max]
    量化会把格子浪费在尾巴上,导致大部分权重精度下降.
    先选一个更紧的阈值 max_val < w_max,把尾巴砍掉,
    再量化,就能把有效格子留给主体分布, 进而减少MSE.
    """
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    """裁剪操作
    参数:
        module: 神经网络模块
        clip_list: 裁剪列表, 包含每个线性层的名称和限制最大值
    """
    from ..utils.module import get_op_by_name

    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        # 使用 torch.clamp 将权重限制在 [-max_val, max_val] 范围内.
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
