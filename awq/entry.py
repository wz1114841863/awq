# 用于对预训练语言模型进行微调后效果的评估,以便了解模型在不同自然语言处理任务上的表现.
from lm_eval import evaluator, tasks

# transformers是hugging Face 提供用于加载和使用各种预训练模型的库.
# AutoModelForCausalLM 是一个通用接口,用于加载因果语言建模
# AutoTokenizer自动加载与模型配套的分词器(Tokenizer),用于将文本转换为模型可理解的输入
# AutoConfig加载模型的配置文件(如层数/隐藏维度等),用于初始化模型或检查模型结构.
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
import tqdm

#  Hugging Face 提供用于简化多设备(CPU/GPU/TPU)和分布式训练的库.
from accelerate import (
    init_empty_weights,  # 用于初始化一个空的模型权重,减少内存占用.
    infer_auto_device_map,  # 自动生成模型的设备分配映射,将模型的不同部分分配到不同的设备上.
    dispatch_model,  # 在多个设备(如 GPU)之间分配模型,提高运行效率.
    load_checkpoint_in_model,  # 加载模型的权重检查到模型中,支持分布式或部分加载.
)
from accelerate.utils.modeling import get_balanced_memory

# datasets 库是Hugging Face提供的一个用于高效处理和使用各种数据集的工具包.
from datasets import load_dataset
from torch import nn

from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model


"""
--model_path:   字符串类型,指定Hugging Face模型的路径.
--batch_size:   整数类型,默认值为1,指定批量大小.
--tasks:        字符串类型,默认值为None,指定任务.
--output_path:  字符串类型,默认值为None,指定输出路径.
--num_fewshot:  整数类型,默认值为0,指定few-shot学习的数量.

--parallel:         布尔类型,启用模型并行.
--max_memory:       字符串类型,可以接受多个参数,指定设备ID与最大内存的映射关系,例如0:10GiB 1:10GiB cpu:30GiB.
--auto_parallel:    布尔类型,自动设置并行和批量大小.

--w_bit:            整数类型,默认值为None,指定权重量化的位数.
--q_group_size:     整数类型,默认值为-1,指定量化组的大小.
--no_zero_point:    布尔类型,禁用零点量化.
--q_backend:        字符串类型,默认值为"fake",选择量化的后端,可选值为"fake"或"real"

--dump_quant:       字符串类型,默认值为None,保存量化后的模型.
--dump_fake:        字符串类型,默认值为None,保存伪量化后的模型.
--load_quant:       字符串类型,默认值为None,加载量化后的模型

--run_awq:          布尔类型,执行AWQ搜索过程.
--dump_awq:         字符串类型,默认值为None,保存AWQ搜索结果.
--load_awq:         字符串类型,默认值为None,加载AWQ搜索结果.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "得到的结果: ['0:10GiB', '1:10GiB', 'cpu:30GiB']"
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument(
    "--dump_fake", type=str, default=None, help="save fake-quantized model"
)
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
parser.add_argument(
    "--vila-20",
    action="store_true",
    help="quantizing or smoothing vila 2.0 (NVILA)",
)
parser.add_argument(
    "--smooth_scale",
    action="store_true",
    help="generate the act scale of visiontower",
)
parser.add_argument(
    "--media_path",
    type=str,
    nargs="+",
    help="The input video to get act scale for visiontower",
)
parser.add_argument(
    "--act_scale_path",
    type=str,
    default=None,
    help="Path to save act scale",
)
args = parser.parse_args()
assert (
    args.act_scale_path is not None and len(args.media_path) > 0
) or not args.smooth_scale

vila_10_quant_mode = (
    ("llava" in args.model_path.lower() or "vila" in args.model_path.lower())
    and not args.vila_15
    and not args.vila_20
)

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)


def build_model_and_enc(model_path):
    "build model and tokenizer"
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        # 加载分词器
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    # 加载模型
    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )  # 初始化类, 用于加载权重

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()  # linear层换为了WQLinear层
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)
                # awq_results.keys(): ["scales", "clip"]
            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    # a hack here to auto set model group
    if args.smooth_scale and args.vila_20:
        if os.path.exists(args.act_scale_path):
            print(f"Found existing Smooth Scales {args.act_scale_path}, skip.")
        else:
            from awq.quantize import get_smooth_scale

            act_scale = get_smooth_scale(args.model_path, args.media_path)
            os.makedirs(os.path.dirname(args.act_scale_path), exist_ok=True)
            torch.save(act_scale, args.act_scale_path)
            print("Save act scales at " + str(args.act_scale_path))
            args.model_path = args.model_path + "/llm"

        if args.dump_awq is None and args.dump_quant is None:
            exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    model, enc = build_model_and_enc(args.model_path)

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048  # 上下文窗口长度(如2048),用于分块处理长文本
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen  # 总样本数 = 总token数 // seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )  # 按seqlen分块处理
                # 输入token: [A, B, C, D]
                # shift_logits: 预测[B, C, D] (基于[A, B, C])
                # shift_labels: 真实标签[B, C, D]
                with torch.no_grad():
                    lm_logits = model(batch).logits  # 获取模型输出的logits
                shift_logits = (
                    lm_logits[:, :-1, :].contiguous().float()
                )  # 去掉最后一个token的logits
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][
                    :, 1:
                ]  # 去掉第一个token的标签
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )  # 计算交叉熵损失
                neg_log_likelihood = loss.float() * model.seqlen  # 当前块的NLL
                nlls.append(neg_log_likelihood)  # 累积所有块
            # 计算困惑度: 对所有块的NLL求和,除以总样本数和seqlen,然后取指数
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
