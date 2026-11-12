# 实验参数与数据协议

## 五类与划分

| 目录类别 | 正常训练 | 正常测试 | 异常测试 |
|---|---:|---:|---:|
| glass-insulator | 2298 | 581 | 90 |
| lightning-rod-suspension | 462 | 117 | 50 |
| polymer-insulator-upper-shackle | 935 | 235 | 102 |
| vari-grip | 477 | 114 | 111 |
| yoke-suspension | 4834 | 1207 | 49 |
| 合计 | 9006 | 2254 | 402 |

仅使用以上五类。训练`train/good`；测试`test/good`及其余缺陷目录。异常为正类。沿用已有目录划分，未新划分独立验证集，也未对来源无人机帧做额外去重审计。相对路径、标签、大小记录在`configs/data_manifest_relative.csv`；50张校准参考另有内容SHA256。其余样本没有逐图内容哈希。

共同输入：完整文件PIL RGB → 双线性Resize((392,392)) → ToTensor → ImageNet均值[0.485,0.456,0.406]及标准差[0.229,0.224,0.225]归一化。float32，AMP关闭，workers=4，seed=1。测试按类别独立组批，先正常后异常，batch=32。

## A5000

配置真值：`configs/a5000.json`；原实际配置：`provenance/a5000/config.json`。

| 参数 | 值 |
|---|---|
| backbone | dinov2reg_vit_base_14；patch14；embed_dim768；4 register tokens |
| 编码器层 | 零基2,3,4,5,6,7,8,9 |
| Bottleneck | 768→256→3072→768；Dropout0.4 |
| Decoder | 8层，12 heads，MLP ratio4，LinearAttention2（la=1） |
| 分组/CAR | lc=2；cr=1 |
| 更新 | Bottleneck/Decoder，从头初始化；编码器冻结 |
| 正式训练步数 | **5000** |
| 学习率曲线尺度 | **15000**，仅运行前5000步，不能改为5000尺度 |
| batch/drop_last | 32/True；shuffle=True |
| 优化器 | StableAdamW |
| lr/nb_lr | 0.002/0.002 |
| betas/eps | (0.9,0.999)/1e-10 |
| weight_decay | 0.01 |
| warmup | 100步 |
| lr_decay_ratio | 0.1 |
| 梯度裁剪 | 全局范数0.1 |
| Loose Loss | p=min(0.9×step/1000,0.9)；factor0.1 |
| 正式评估 | 2500和5000步，表格取固定最终5000步 |
| 状态备份 | 每1000步及评估点覆盖last.pt；评估点另存轻量权重 |
| 评分 | 原392误差图→256→5×5 sigma4高斯→最高655像素均值 |

原训练的首批索引和RNG摘要保留在初始化记录中；这能检测初始化/采样流程偏离，但不能保证跨硬件逐位相同。

## NCut

`configs/a5000.json`中的`ncut`保存全部参数。

| 参数 | 值 |
|---|---|
| 模型/额外训练 | 同一个A5000 / 0步 |
| 特征层 | encoder.blocks[9]，norm后去CLS和4 registers |
| 特征网格 | 28×28 |
| 正常参考 | 固定50张，每类10张train/good |
| tau来源 | 参考图非对角余弦相似度的0.75分位数 |
| 原tau | 0.47686619662747487 |
| 边权 | 相似度>tau为1，否则1e-5 |
| 切分 | 第二广义特征向量的均值 |
| 自动选区 | 最大绝对特征值位置所在分区的4连通分量 |
| 面积回退 | <0.01或>0.99回退全图 |
| 图密度回退 | <0.005或>0.995 |
| 最小谱间隙 | 1e-8 |
| 膨胀/闭运算 | 都关闭 |
| 区域到评分图 | 最近邻缩放到256×256 |
| 区域K | min(655,区域像素数) |
| 融合 | 0.5全图**分数**＋0.5区域**分数** |

## CostFilter

配置真值：`configs/costfilter.json`；原实际配置：`provenance/costfilter/config.json`。

| 参数 | 值 |
|---|---|
| 基础模型 | 完整A5000冻结且eval |
| 可训练模块 | 原CostFilter 3D过滤网络；base_channels=48 |
| 训练步数 | **额外5000**步；约6.66轮，不是原配方17 epochs |
| batch/eval_batch/workers | 12/32/4 |
| 优化器 | Adam，lr1e-4，betas(0.9,0.98) |
| LR调度 | ReduceLROnPlateau；factor0.5；patience5；按完整epoch平均训练loss更新 |
| 梯度裁剪 | 最大范数2.0 |
| 精度 | float32，无AMP |
| 代价体 | B×768×2×64×64 |
| 输入特征 | 两组B×768×28×28 |
| 引导误差 | 28→64，训练和推理一致 |
| 分类头 | 5类资产 |
| 额外数据 | DTD47类5640张；正常图片合成像素标签 |
| Perlin | 448×448，周期2^randint(0,6)，阈值0.3后最近邻缩放392 |
| self_shuffle | 392上8×8块，每块49×49 |
| 融合 | **图**：0.95原始误差图＋0.05过滤概率图 |
| 后处理 | 融合392图→256→同一高斯→全图最高655像素均值 |
| checkpoint | 每1000步覆盖last.pt，最终另存final.pt |
| 测试选择 | 固定最终5000步，仅最终正式评估；系数0.05沿用上游默认 |

原运行日志记录过滤网络训练1531.86秒、最终评估46.50秒；这只是当时设备/环境实测，不是新服务器耗时保证。

## 指标范围

I-AUROC是每类图像级ROC面积。主表为五类等权平均。没有真实像素标注，因此不报告P-AUROC。FPR@95TPR使用各类别测试ROC上首个TPR≥95%的工作点，是诊断指标，不是独立校准的部署阈值。

宏平均FPR@95TPR：A43.47%、NCut35.75%、CostFilter43.66%。NCut并非每类都获益：Vari-grip AUROC从87.27%降至86.72%。本轮单seed结果没有多次独立训练的方差或显著性检验。
