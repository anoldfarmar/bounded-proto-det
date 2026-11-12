# InsPLAD完整视野：Dinomaly2、NCut与CostFilter对照

本仓库封装同一套五类InsPLAD划分上的三个实验，提供可执行源码、完整参数、原始逐图分数与日志。**图片、预训练权重、A5000权重、CostFilter权重和Conda环境均不复制进仓库**，通过一个路径配置引用已有文件。

| 类别 | A5000全图 | NCut＋全图融合 | CostFilter＋全图融合 |
|---|---:|---:|---:|
| Glass Insulator | 90.77% | **92.08%** | 91.77% |
| Lightning Rod Suspension | 81.85% | **88.75%** | 82.92% |
| Polymer Insulator Upper Shackle | 87.46% | **90.53%** | 87.89% |
| Vari-grip | 87.27% | 86.72% | **88.83%** |
| Yoke Suspension | 79.30% | **84.69%** | 80.55% |
| 五类等权平均I-AUROC | 85.33% | **88.56%** | 86.39% |

这些是本地一次seed=1实验的结果，不是原论文在InsPLAD上的公开成绩。平均值先对每类计算AUROC再取算术平均，不是把2656张测试图混在一起计算。

## 三个实验分别改了什么？

| 实验 | 训练内容 | 推理方法 | 额外数据 |
|---|---|---|---|
| A5000 | 冻结DINOv2；从头训练Dinomaly2 Bottleneck/Decoder 5000步 | 全图重建误差最高655个像素的均值 | 无 |
| NCut＋全图融合 | 复用A5000；额外训练0步 | 同一误差图，NCut区域分数与全图分数各占0.5 | 50张正常训练参考图用于图阈值校准，已包含在原训练集 |
| CostFilter＋全图融合 | 冻结整个A5000；新增CostFilter训练5000步 | 先以0.95/0.05融合重建误差图与过滤概率图，再取Top-655均值 | DTD纹理、合成异常像素监督、五类资产类别标签 |

**两个“融合”不等价：NCut在图像分数层面融合；CostFilter在像素图层面融合后重新选择Top-K。** CostFilter没有叠加NCut。它是Dinomaly2＋CostFilter的适配比较，原CostFilter仓库提供的对应基础模型是Dinomaly。

完整解释见[方法与代码对应](docs/METHODS.md)、[参数及协议](docs/PARAMETERS.md)、[来源与适配](docs/REFERENCES.md)。

## 先阅读和核对结果

```bash
cd /root/ADAD/InsPLAD-ThreeMethods
python scripts/verify_results.py
```

此命令只需Python 3.11标准库，在Windows也能运行；根据2656张逐图CSV重新计算三种方法的逐类AUROC、宏平均和FPR@95TPR，并核对相同基线分数与NCut融合算术。无需数据、权重或GPU。`--files`额外校验交付文件哈希。

- [统一精确结果](results/summary.json)、[逐类比较CSV](results/comparison.csv)。
- `results/recorded/`：三套逐图分数、原始summary与metrics。
- `provenance/`：原配置、日志、源文件哈希、输入协议和NCut正常参考校准记录。

## 配置外部路径

```bash
cp configs/paths.example.json configs/paths.local.json
# 编辑 configs/paths.local.json，填写自己的数据、三份权重和输出目录。
```

示例已填写当前服务器的有效路径。需要的三份权重分别是：DINOv2 ViT-B/14 reg4预训练参数、Dinomaly2 A5000最终参数、CostFilter最终参数。文件SHA256见[资产身份](docs/ASSETS.md)。外部数据包括InsPLAD五类；**只有训练CostFilter才需要DTD**，仅评估已有过滤网络不需要DTD。

所有运行入口在自身进程内绕过代理，并离线加载本地权重，不自动下载任何文件。若换服务器，指定两个已有环境的解释器：

```bash
export PYTHON_BIN=/root/autodl-tmp/envs/dinomaly2/bin/python
export COSTFILTER_PYTHON_BIN=/root/autodl-tmp/envs/costfilter_ad/bin/python
```

A/NCut原环境NumPy为2.2.6，CostFilter因imgaug使用NumPy1.26.4；两个环境分别保留。详细版本和可选安装指令在[环境说明](docs/ENVIRONMENT.md)。

## 用已有权重复现评估

```bash
# CPU检查：确认划分、预训练编码器、A5000和50张正常参考身份，不训练。
bash scripts/run.sh check

# 同一A5000，正式评估2656张，同时输出全图、NCut区域和NCut融合分数。
bash scripts/run.sh ncut-evaluate

# 正式评估已有CostFilter，输出A5000、过滤网络和0.05融合分数。
bash scripts/run.sh costfilter-evaluate
```

`ncut-evaluate`默认使用原正常参考校准出的τ=0.47686619662747487。加`--recalibrate`会在原50张正常训练参考图上重新校准；不使用测试标签选区。正式复现会检查A与NCut分数对原始记录的误差；CostFilter也会检查A5000分数的一致性。

仅检查执行可加`--smoke`：

```bash
bash scripts/run.sh ncut-evaluate --smoke
bash scripts/run.sh costfilter-evaluate --smoke
```

这两个命令只评估每类固定一张正常和一张异常，共10张；小批量数值可能不同，**不能将smoke指标写成实验结果**。

## 从头训练复现

```bash
# 实验1：重新训练A5000，5000步后停止。中途2500步和最终5000步评估。
bash scripts/run.sh a-train --train-only

# 定位刚完成的新A5000目录；输出根目录从paths配置读取。
A_RUN="$(python scripts/latest_run.py a5000)"

# 实验2：该新A5000的NCut评分，额外训练0步。
bash scripts/run.sh ncut-evaluate --baseline-run "$A_RUN" --recalibrate

# 实验3：冻结该新A5000，训练CostFilter 5000步并最终评估。
bash scripts/run.sh costfilter-train --baseline-run "$A_RUN"
```

若复用原A5000，只需`bash scripts/run.sh costfilter-train`。训练前可执行`costfilter-check`检查DTD及正常图合成预览。`costfilter-smoke`会训练2步，仅作执行检查。CostFilter恢复训练和重新评估：

```bash
bash scripts/run.sh costfilter-train --resume /你的新过滤网络运行目录
bash scripts/run.sh costfilter-evaluate --run /你的新过滤网络运行目录
```

加`--paths /另一份路径配置.json`可切换服务器路径，避免改动算法文件。A训练及NCut默认输出到配置中`output_root/a5000/`、`output_root/ncut/`；CostFilter输出到`output_root/costfilter/`。新运行目录按时间和PID建立，原始封存结果保持独立。

从头训练遵守原初始化、采样及超参数；不同硬件/库版本的浮点行为可能使新权重与指标变化。复用原权重重新评估和重新训练是两种复现范围，验证记录见[VALIDATION.md](VALIDATION.md)。

## 仓库结构

```text
configs/      A5000与CostFilter参数、外部路径样例、相对数据清单、50张正常参考身份
scripts/      可移动的训练/评估入口、CPU检查和标准库结果重算
vendor/       实验使用的Dinomaly2与CostFilter源码；无需另外克隆上游仓库
docs/         方法、参数、来源、资产、环境及上游README
results/      三个实验的封存结果及逐图分数
provenance/   原日志、配置、哈希、校准证据及打包适配差异
environment/  两套环境的版本记录与运行依赖
tests/        复现契约检查
```

## 方法来源与实验含义

基础模型来自[Dinomaly2](https://github.com/guojiajeremy/Dinomaly2)，图切分参考[TokenCut](https://github.com/YangtaoWANG95/TokenCut)，匹配代价过滤来自[CostFilter-AD](https://github.com/ZHE-SAPI/CostFilter-AD)。本仓库记录我们在五类InsPLAD完整视野条件下的适配；来源、保留文件和改动逐项见[REFERENCES.md](docs/REFERENCES.md)。本地原项目来自源码包，没有Git历史，因此以实验记录的文件SHA256定位版本，不虚构上游commit。

NCut主要观察到的收益是在约95%异常召回率下减少正常误报。候选区不是人工前景真值，背景机制仍需结合失败样本审核；单seed探索结果不构成独立保留集上的SOTA结论。CostFilter引入额外训练和DTD监督，其成本不同，不能描述为完全同等训练预算。
