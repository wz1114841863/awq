# 执行步骤
## 执行 AWQ 搜索并保存搜索结果
python -m awq.entry --model_path facebook/opt-125m --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/opt-125m-w4-g128.pt

## 在 WikiText-2 上评估 AWQ 量化模型(模拟伪量化)
python -m awq.entry --model_path facebook/opt-125m --tasks wikitext --w_bit 4 --q_group_size 128 --load_awq awq_cache/opt-125m-w4-g128.pt --q_backend fake

## 生成真实量化权重 (INT4)
python -m awq.entry --model_path facebook/opt-125m --w_bit 4 --q_group_size 128 --load_awq awq_cache/opt-125m-w4-g128.pt --q_backend real --dump_quant quant_cache/opt-125m-w4-g128-awq.pt

## 加载并评估真实量化模型，使用AWQ提供的内核
python -m awq.entry --model_path facebook/opt-125m --task wikitext --w_bit 4 --q_group_size 128 --load_quant ./quant_cache/opt-125m-w4-g128-awq-v2.pt
