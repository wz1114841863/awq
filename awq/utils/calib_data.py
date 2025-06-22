import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    """ 从数据集加载文本数据, 进行处理后生成用于校准的样本集合 """
    if data == "pileval":
        # 目前仅支持"pileval" 数据集
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    # 将数据集按照随机种子打乱, 以确保后续取样的随机性和可复现性.
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:  # len(dataset) = 214670
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            # 如果编码后的长度超过 512, 则跳过该样本, 不能超过设定输入
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)  # samples: [n_samples, len(text)]
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [  # return: [n_split, block_size]
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
