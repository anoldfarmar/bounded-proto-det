# 参考仓库、源码身份与适配边界

| 来源 | 本实验使用的部分 | 链接 |
|---|---|---|
| Dinomaly2 | 冻结ViT、Noisy Bottleneck、Decoder、CAR、Loose Loss、原评分工具 | https://github.com/guojiajeremy/Dinomaly2 ，论文 https://arxiv.org/html/2510.17611v3 |
| DINOv2 | ViT-B/14及register版本编码器；本地预训练权重 | https://github.com/facebookresearch/dinov2 |
| TokenCut | 特征相似图、归一化切分、自动种子连通分量的思路 | https://github.com/YangtaoWANG95/TokenCut ，论文 https://arxiv.org/abs/2202.11539 |
| CostFilter-AD | Costfilter_Dinomaly的3D过滤网络、组合损失、合成异常和代价体构造参考 | https://github.com/ZHE-SAPI/CostFilter-AD ，论文 https://arxiv.org/abs/2505.01476 |
| Dinomaly | 原CostFilter对应分支的基础模型来源；本实验实际使用Dinomaly2 | https://github.com/guojiajeremy/Dinomaly |
| DTD | 训练CostFilter合成异常所用纹理 | https://www.robots.ox.ac.uk/~vgg/data/dtd/ |

上游链接用于引用，不在运行时访问。当前实验采用本地源码包快照，原目录没有Git历史；网页最新版本可能与当时快照不同。**实验版本以本仓库源码和原记录的SHA256确定**，不是重新拉取上游main分支。

## 本地来源

- Dinomaly2原目录：`/root/ADAD/Dinomaly2-main`。本仓库`vendor/dinomaly2/`取自已封装A5000/NCut的运行时源码，并与原训练记录24个文件的哈希核对。
- CostFilter原目录：`/root/ADAD/CostFilter-AD-main`，由用户提供的源码包解压。网络、loss、Perlin、数据适配及代价体/损失适配函数与原2026-11-06运行快照逐字节核对；见`provenance/source_identity.json`及原`source_sha256.json`。
- `docs/*UPSTREAM_README.md`保留本地上游说明；`COSTFILTER_DINOMALY_README.md`保留对应分支说明。

## 这轮实验之前已经做的算法/数据适配

1. InsPLAD五类白名单；支持JPG等原数据扩展名；同一训练/测试划分。
2. 完整文件缩放392；替代此前448→CenterCrop392。三种方法采用一致视野。
3. NCut使用DINOv2中间层norm后patch tokens，正常参考校准图阈值；属于TokenCut风格适配，未原样复现TokenCut论文的特征/默认阈值。
4. CostFilter接到Dinomaly2的CAR分组目标与重建，而非原Dinomaly模型；五类分类头。
5. CostFilter训练预算5000步，非上游17epochs。增加DTD、合成像素监督和资产类别标签。
6. Perlin在448噪声画布上生成，再将二值mask最近邻缩放392；不裁剪输入。self_shuffle为392上的8×8格。
7. DTD先在uint8上做imgaug再转float；SSIM支持明确设备；Focal微小截断避免饱和NaN；调度器按epoch平均loss更新。
8. 原CostFilter训练引导误差为28→64，测试为28→392→64；本实验统一28→64。
9. CostFilter评分统一采用256网格、同一高斯及Top1%，与A/NCut对齐；融合系数0.05保留上游默认。

## 此次封装额外做的工程适配

- `scripts/`将数据、权重和输出路径集中配置。训练/评分调用记录的数值实现，未重新设计CAR、NCut或CostFilter。
- 数据身份校验改为相对路径/标签/大小比较，允许数据挂载点变化；50张正常参考仍验内容SHA256。
- CostFilter参考分数通过`baseline_reference/`桥接到当前图片路径。此目录只保存评分，不冒充完整训练目录。
- `vendor/costfilter/.../run.py::evaluate`只允许检查点配置中的七个纯路径字段变化；**其他超参数、baseline哈希、精度及smoke身份继续逐项一致**。这是为已有权重跨路径评估做的唯一原runner修改；原文件保留于`provenance/original_costfilter_runner.py`，差异见`portability.patch`。
- 原run.sh和vendor内原绝对路径配置仅用于追溯。移动仓库后使用根README的`scripts/run.sh`。

## 版权与引用

保留各来源代码中的原文件头、版权与许可证信息。Dinomaly2本地README声明Apache-2.0，但源码包没有顶层LICENSE；CostFilter本地LICENSE为MIT，已原样保留在`vendor/costfilter/LICENSE`。本整理仓库不额外声明一个覆盖所有第三方代码的统一许可证，也不将上游模块视为本人的原创。

NCut分支的研究贡献应围绕当前场景下的问题验证、评分设计及证据展开；不能仅将TokenCut本身或现成CostFilter网络作为新的原创算法。
