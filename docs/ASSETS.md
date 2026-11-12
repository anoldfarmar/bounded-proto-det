# 外部资产与哈希

本仓库没有权重或图片副本。修改`configs/paths.local.json`即可引用已有文件。

| 资产 | 当前外部路径 | SHA256 |
|---|---|---|
| DINOv2预训练编码器 | `/root/autodl-tmp/model/dinov2_vitb14_reg4/dinov2_vitb14_reg4_pretrain.pth` | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| A5000 | `/root/autodl-tmp/experiments/insplad_fullview_B/runs/train_seed1_20261102_104902_22370/results/checkpoints/step_005000.pt` | `03cc73ef4cf9f9c019b9740ed06ac2cb578624886e62506e9a5c76820aae2941` |
| CostFilter5000 | `/root/autodl-tmp/experiments/insplad_fullview_costfilter/runs/train_seed1_20261106_140759_10780/checkpoints/final.pt` | `b7912bb140756bb3012dfb8af301898af246c94835be9ec8626bcd076025fab4` |

InsPLAD：`/root/autodl-tmp/data/insplad/unsupervised_anomaly_detection`。DTD训练纹理：`/root/autodl-tmp/data/dtd/dtd/images`，47类5640张。

原始运行目录和配置以`provenance/`记录为准。A原运行的NCut分支：`ncut/step_005000_20261102_115421_22370`。

只核对封存CSV：不需任何资产。仅评估A/NCut：InsPLAD、编码器、A5000。评估已有CostFilter：再需过滤网络权重，**不需DTD**。从头训练A：InsPLAD和预训练编码器；新A权重通过`--baseline-run`交给后续分支。从头训练CostFilter：还需DTD和已完成A5000，过滤网络随机初始化。

原`dataset_manifest_sha256`包含旧绝对路径，移动后该旧哈希不能直接作为新挂载点的校验。本入口核对同一相对清单。DTD仅校验数量、类别数和图片完整性，未保存全部纹理的内容哈希。
