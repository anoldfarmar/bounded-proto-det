# InsPLAD 五类：Dinomaly2 A5000 + CostFilter

这是用户选定的 **Dinomaly2＋CostFilter 适配实验**。复用已训练好的完整视野 A5000，冻结整个 Dinomaly2，仅训练 CostFilter。它不是原仓库 Dinomaly＋CostFilter 的原封不动复现，也没有将 NCut 叠加到 CostFilter。

已有对照：A5000 全图 85.33%；NCut 区域 88.28%；NCut＋全图融合 88.56%。这次的目的，是在相同五类、划分和输入视野上比较另一种后处理网络。额外的训练数据和计算必须披露，不能宣称与 NCut 等训练成本。

## 固定数据与输入

| 类别 | 正常训练 | 正常测试 | 异常测试 |
|---|---:|---:|---:|
| glass-insulator | 2298 | 581 | 90 |
| lightning-rod-suspension | 462 | 117 | 50 |
| polymer-insulator-upper-shackle | 935 | 235 | 102 |
| vari-grip | 477 | 114 | 111 |
| yoke-suspension | 4834 | 1207 | 49 |
| 合计 | 9006 | 2254 | 402 |

- InsPLAD：`/root/autodl-tmp/data/insplad/unsupervised_anomaly_detection`。
- DTD：`/root/autodl-tmp/data/dtd/dtd/images`，47类、5640张纹理图。
- 输入：当前文件的**完整视野**双线性缩放为392×392；不裁剪、不补边、不人工调整前景方向。
- ImageNet归一化，和原A5000同样的变换顺序。原始宽高比会变成正方形，与现有完整视野对照一致。
- DINOv2 ViT-B/14 reg4 和 A5000 检查点均只读，启动前校验SHA256；模型来源详见 `config.json`。
- 不读真实像素级异常标注，不计算或虚构 P-AUROC。

## 接入流程和源码位置

```text
完整 InsPLAD 图像392×392
    ↓
冻结 Dinomaly2 A5000：编码器 → CAR目标 / Bottleneck → Decoder
    ↓ 两组目标与重建特征，各为 B×768×28×28
model_adapter.cost_inputs
    ├─ 计算每个位置的全局匹配代价，保留768个候选
    ├─ 组装 B×768×2×64×64 代价体
    ├─ 提供两组 B×784×768 目标特征作引导
    └─ 提供两通道的初始重建误差图作引导
    ↓
原仓库 CostFilter 3D 网络（base_channels=48）
    ├─ 正常/异常像素 logits
    └─ 五类资产分类 logits
    ↓
过滤概率图，以及固定0.05比例的融合图
```

- `insplad_data.py`：五类白名单、完整视野读取、DTD/结构打乱合成。
- `model_adapter.py`：加载冻结A5000、匹配代价体、原网络初始化和组合损失。
- `run.py`：数据校验、训练、原子检查点、恢复训练、评估、日志和热图。
- `config.json`：参数和所有输入/输出路径。
- `Costfilter_Dinomaly/unet3d_att_dino_channel_test_48_min.py`：原版过滤网络，只新增可选 `cls_classes` 参数，默认15保持原入口兼容，本实验传5。
- `Costfilter_Dinomaly/loss.py`：复用原损失，将SSIM窗口改为随设备迁移的buffer，移除构造时强制`.cuda()`。

CostFilter 的引导特征来自 Dinomaly2 返回的 CAR 目标。它们不能称为未经CAR的原始冻结编码器特征。这里也没有生成新的语义前景掩码，不能仅凭分数提升就断言背景已被正确分割。

## 训练参数

| 参数 | 本实验 |
|---|---|
| 新训练模块 | 仅CostFilter；整个A5000冻结并保持eval模式 |
| 步数 | 5000次过滤网络更新；不重新训练5000步Dinomaly2 |
| batch | 12，原CostFilter训练入口的设置 |
| 测试batch | 32，按类别分别组批、先正常后异常，与原A5000完全一致 |
| workers | 4 |
| 精度 | float32，不启用AMP |
| 优化器 | Adam，lr=0.0001，betas=(0.9,0.98) |
| 梯度裁剪 | 最大范数2 |
| LR调度 | ReduceLROnPlateau，factor=0.5，patience=5；每个完整epoch按平均训练loss更新 |
| 随机种子 | 1；每个样本按(seed,epoch,index)生成独立合成随机数 |
| checkpoint | 每1000步覆盖`last.pt`；最后另存`final.pt` |
| 测试评估 | 完成后一次完整评估；不用测试AUROC选择检查点或融合比例 |

原仓库过滤网络训练17个epoch，本实验先采用5000步预算（约6.66轮五类数据）。这是明确的预算调整，不能把结果标为已完整执行原17轮配方。实际训练时长以日志前50–100步的ETA为准；首次初始化会让前几步偏慢。

合成规则沿用上游的两种素材来源：约一半DTD纹理，一半原正常图增强后8×8分块打乱。再以约0.5概率保留原正常图，其余样本通过Perlin掩码混合素材。合成像素标签自动生成，**不用InsPLAD真实异常图或人工缺陷标签训练**。类别标签取自五个资产目录，是额外的资产类别监督；不能把它称为完全不使用任何标签。

## 为完整视野及现代环境做的修改

1. 支持`.jpg/.jpeg/.png`等大小写扩展名，按白名单读取五类。
2. 图像直接完整缩放392；原版448→CenterCrop392被替换。
3. 原Perlin函数要求尺寸能被2到32的周期整除，392不满足。保持448噪声画布，阈值化后以最近邻缩放掩码到392；**不裁剪图像**。结构打乱在392上使用8×8格，每格49像素。
4. DTD在uint8上执行imgaug，再转float混合。上游先转float32，会被Autocontrast等PIL增强拒绝。
5. 保留上游代价体方向（decoder query、encoder reference）和top768后按原候选索引排序的规则。CPU测试验证与原版逐元素一致，覆盖batch=1。
6. 上游训练引导误差走28→64，推理走28→392→64。本实验统一28→64，避免训练/推理不一致；原A5000评分图仍按28→392生成。
7. 五类分类头；移除新版PyTorch不再支持的scheduler `verbose` 参数；SSIM支持CPU和显式GPU设备。
8. Focal loss输入做极小概率截断，避免softmax饱和时`(1-pt)^gamma`的分数幂产生NaN。
9. 学习率调度按完整epoch的平均训练loss更新，替代上游最后一个batch的loss。

## 公平评分方式

在同一批测试图上记录三个方法：

| 输出名 | 定义 |
|---|---|
| `A5000_full` | 原始A5000重建误差图 |
| `costfilter_only` | CostFilter异常类别概率图 |
| `costfilter_fused_0p05` | `0.95×A5000误差图 + 0.05×过滤概率图`；融合系数来自原仓库默认值，未按测试集调整 |

三者都先生成392图，再缩放256，应用原A5000的5×5、sigma=4高斯核，取最高655个像素的均值（全图1%）。报告五类各自I-AUROC及算术平均，不以所有样本混合AUROC替代类别宏平均。

上游CostFilter测试默认392评分；本实验统一到已有A5000/NCut的256评分协议，因此属于对齐后的方法比较。每次评估会逐图核对原A5000已保存分数，最大绝对误差超过1e-4则判定对照失配。

I-F1和FPR@95TPR作为测试集诊断量，不表示已有独立部署阈值。沿用之前探索过的测试集，不能据此声称独立保留集上的SOTA。

## 运行指令

所有命令均使用 `/root/autodl-tmp/envs/costfilter_ad/bin/python`。`run.sh`仅在自身进程内绕过代理，不下载模型、包或DTD。

```bash
conda activate /root/autodl-tmp/envs/costfilter_ad
cd /root/ADAD/CostFilter-AD-main

# 无卡也可做：校验文件、数据数量和合成预览；不训练
bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh check

# 可选：仅2次更新和10张固定测试图，验证执行链路
bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh smoke

# 正式：5000步过滤网络训练，完成后自动评估全部2656张测试图
bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh train
```

后台正式运行：

```bash
cd /root/ADAD/CostFilter-AD-main
mkdir -p /root/autodl-tmp/experiments/insplad_fullview_costfilter
nohup bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh train \
  > /root/autodl-tmp/experiments/insplad_fullview_costfilter/console.log 2>&1 &
echo "PID=$!"
```

查看日志：

```bash
tail -f /root/autodl-tmp/experiments/insplad_fullview_costfilter/console.log
```

断点续训（把路径替换为实际运行目录）：

```bash
bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh train \
  --resume /root/autodl-tmp/experiments/insplad_fullview_costfilter/runs/train_seed1_时间_PID
```

恢复时重新验证数据、模型、源文件和环境，恢复优化器/LR/随机状态以及epoch内的下一批索引。合成样本不依赖worker调度。Smoke检查点不能作为正式实验恢复。

单独重做最新一次正式运行的最终评估：

```bash
bash experiments/insplad_fullview_dinomaly2_costfilter/run.sh evaluate
```

## 输出位置与检查点体积

```text
/root/autodl-tmp/experiments/insplad_fullview_costfilter/
├── latest_train.txt          最新正式运行目录
├── console.log               用户nohup命令的控制台日志
├── check/                    无GPU检查和训练合成预览
├── smoke/                    两步验证，不是正式结果
├── validation/               原A5000只读一致性检查
└── runs/train_seed1_时间_PID/
    ├── config.json / identity.json / environment.txt
    ├── source/ / source_sha256.json
    ├── dataset_manifest.csv / dtd_manifest.csv / data_check.json
    ├── log.txt / run_status.json / timing.json
    ├── synthetic_preview/index.html
    ├── checkpoints/last.pt    最新可恢复状态，包含Adam状态
    ├── checkpoints/final.pt   最终过滤网络权重
    ├── summary.json           最终指标，主要反馈文件
    ├── per_category_metrics.csv
    ├── image_scores.csv       逐图三个分数及原A分数核对
    ├── REPORT.md              可读结果表格
    └── visualizations/index.html
```

只保存过滤网络，A5000和编码器用校验后的本地文件加载，不重复保存。`last.pt`约0.5GiB、`final.pt`约0.16GiB；原子写入时临时文件会额外占用空间。目录中的`run_status.json`应为`complete`，且`summary.json`中`smoke=false`、`filter_steps=5000`、`test_image_count=2656`才是正式完成。

可视化每类固定首张正常图和首张异常图，用于说明原图、A5000、过滤概率、融合图的差别。A和融合热图共用色标，过滤概率单独使用0–1色标。没有人工标注任务，热图也不是工件前景真值。

本目录不改写原Dinomaly2代码、原A5000权重、InsPLAD图片或既有NCut结果。用户提供的`Dinomaly-master.zip`在这条Dinomaly2适配路线中不需要额外解压或训练。
