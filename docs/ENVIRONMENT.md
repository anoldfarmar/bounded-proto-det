# 两套环境

原两套环境都使用Python3.11.17、torch2.9.0+cu128、torchvision0.24.0+cu128。GPU实测为RTX4080SUPER，驱动报告总显存32760MiB。2D实验无需Open3D；没有xFormers时会提示警告，原实验也未使用它。

| 用途 | Conda环境前缀 | NumPy | OpenCV |
|---|---|---|---|
| A5000/NCut | `/root/autodl-tmp/envs/dinomaly2` | 2.2.6 | opencv-python-headless4.12.0.88 |
| CostFilter | `/root/autodl-tmp/envs/costfilter_ad` | 1.26.4 | opencv-python4.11.0.86 |

CostFilter的imgaug0.4.0依赖旧NumPy接口，不能直接套用A环境的NumPy2。不要在原A环境里降级以求统一；入口分别选择解释器。

- A环境关键版本：`environment/dinomaly2_packages.json`、`environment/dinomaly2_requirements-runtime.txt`。
- CostFilter原完整pip记录：`environment/costfilter_pip_freeze.txt`，第一行为Python版本。
- CostFilter运行依赖：`environment/costfilter_requirements-runtime.txt`。它是复现需要的直接依赖记录，不是离线wheel包。

已有环境无需安装。新服务器可在数据盘创建环境并按记录安装，以下命令仅供手动执行，不由复现入口自动运行：

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PIP_PROXY
export NO_PROXY='*' no_proxy='*' PIP_CONFIG_FILE=/dev/null
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/pip-cache /root/autodl-tmp/conda-pkgs
export TMPDIR=/root/autodl-tmp/tmp PIP_CACHE_DIR=/root/autodl-tmp/pip-cache CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs

conda create -p /root/autodl-tmp/envs/dinomaly2 python=3.11 -y
conda activate /root/autodl-tmp/envs/dinomaly2
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/dinomaly2_requirements-runtime.txt
python -m pip check

conda create -p /root/autodl-tmp/envs/costfilter_ad python=3.11 -y
conda activate /root/autodl-tmp/envs/costfilter_ad
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/costfilter_requirements-runtime.txt
python -m pip check
```

这些安装需要网络；CUDA版PyTorch可以在无卡模式安装，实际GPU检查在有卡模式运行。Conda渠道及pip源可按所在网络配置，直接联网不可达时也不会偷偷启用代理。

从头训练想尽量匹配原结果，应保持原Python/PyTorch/timm、NumPy/SciPy等版本与数值环境。新GPU不保证逐位一致。Conda未打包，xFormers未列为必需依赖。
