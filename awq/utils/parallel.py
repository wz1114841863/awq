import os


def auto_parallel(args):
    """ 根据模型大小和设备可用性自动配置并行计算所需的 GPU 数量,
        以及设置环境变量 CUDA_VISIBLE_DEVICES 来控制使用的 GPU """
    model_size = args.model_path.split("-")[-1]
    if model_size.endswith("m"):
        model_gb = 1
    else:
        model_gb = float(model_size[:-1])
    if model_gb < 20:
        n_gpu = 1
    elif model_gb < 50:
        n_gpu = 4
    else:
        n_gpu = 8
    args.parallel = n_gpu > 1
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if isinstance(cuda_visible_devices, str):
        cuda_visible_devices = cuda_visible_devices.split(",")
    else:
        cuda_visible_devices = list(range(8))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(dev) for dev in cuda_visible_devices[:n_gpu]]
    )
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    return cuda_visible_devices
