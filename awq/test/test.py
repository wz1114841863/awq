import gc
import torch
import torch.nn as nn


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    """计算权重缩放因子"""
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)  # [n_group, q_group_size]
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale  # weight.shape[-1]


if __name__ == "__main__":
    # 测试get_weight_scale
    fc_layer = nn.Linear(in_features=128, out_features=64)
    weight = fc_layer.weight
    print(f"weight.shape: {weight.shape}")
    q_group_size = 32
    weight_scale = get_weight_scale(weight, q_group_size)

    print("权重缩放因子形状:", weight_scale.shape)
