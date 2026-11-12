# 完整视野 Dinomaly2-B：新训 A5000，随后 NCut 评分

这是独立的新条件：训练、测试都将整张数据文件直接缩放392×392，无裁剪、无补边。
使用指定五类，9006张正常训练图、2656张测试图；其余12类不使用。
“完整视野”指当前数据文件，非方形图仍会方形缩放。

代码在`experiments/insplad_fullview/`；结果在独立数据盘目录：
`/root/autodl-tmp/experiments/insplad_fullview_B/`。
不覆盖旧`insplad5`实验，不加载旧A/C1训练检查点，不复用旧候选掩码。

## A5000 和 NCut 的关系

A5000是训练5000步的Dinomaly2模型，DINOv2冻结，Bottleneck/Decoder从头初始化并训练。
NCut没有可训练参数，“NCut5000”准确说法是：**对A的5000步检查点增加NCut区域评分**。
主结果为全图分数与NCut种子区域分数各0.5融合；中心选区仅用于诊断。
它们共享一个训练模型，不需要同时训练两份；同一次NCut评估共享前向特征与异常图。
仅显存有余量不能判断GPU计算空闲，并行复制训练不保证更快。

## 参数与配对条件

| 参数 | 设置 |
|---|---|
| 输入 | 整图Resize392×392；无Crop、无Padding |
| 训练停止步数 | **5000** |
| batch / workers / seed | **32 / 4 / 1** |
| 学习率 | 0.002；100步warmup |
| LR调度尺度 | **15000**，保留旧A训练前5000步的LR曲线；实际不会训练15000步 |
| 评估与轻量权重 | 2500、5000步；固定最终5000步，不挑测试最优 |
| 完整备份 | 每1000步及评估前覆盖`last.pt`，原子替换 |
| 优化器 | 原StableAdamW，betas=(.9,.999)、eps=1e-10、weight_decay=.01 |
| Loose loss / CAR / decoder | 沿用旧A；dropout=.4，梯度裁剪=.1；不加前景损失 |
| 混合精度 | 不启用，保持旧A实际运行设置 |
| NCut | 第10层冻结DINOv2 patch特征；固定50张正常训练参考的75%分位数校准阈值 |
| 评分 | 256×256异常图，最高655像素；区域不足则用区域全部 |

LR调度尺度和weight_decay沿用旧A实际训练入口，以免视野之外再引入优化策略变化。
首批随机训练索引和初始化前迭代RNG会与旧A已记录条件核对；后续不保证位级一致。
NCut校准只读取正常训练参考图；测试图不参与阈值拟合或参数选择。
实现为TokenCut风格的DINOv2适配，不是TokenCut原论文整套复现。

## 正式运行（推荐一条命令）

```bash
cd /root/ADAD/Dinomaly2-main
mkdir -p /root/autodl-tmp/experiments/insplad_fullview_B
nohup bash experiments/insplad_fullview/run.sh train \
  > /root/autodl-tmp/experiments/insplad_fullview_B/console.log 2>&1 &
echo "PID=$!"
```

`train`顺序执行：新训A5000 → 保存最终检查点及全图指标 → 同一检查点NCut评估 → 对照汇总。
NCut评估不再训练5000步。已有环境自动使用，代理强制绕过，本地模型离线加载。
不要同时启动`train`与`train-only`；二者都会从头训练。

```bash
# 实时进度；Ctrl+C只退出看日志
tail -f /root/autodl-tmp/experiments/insplad_fullview_B/console.log

# 当前/最近训练目录
cat /root/autodl-tmp/experiments/insplad_fullview_B/LATEST_TRAIN.txt

# 完成后，A结果目录与NCut结果目录
cat /root/autodl-tmp/experiments/insplad_fullview_B/LATEST_A_RESULTS.txt
cat /root/autodl-tmp/experiments/insplad_fullview_B/LATEST_NCUT.txt
```

## 其他模式

```bash
# 无卡可检查：五类图像完整性、来源哈希、输入边缘保留、NCut数学、CPU模型严格加载
bash experiments/insplad_fullview/run.sh check

# 真实数据2步训练+10张测试+NCut+检查点回读；不作为正式性能
bash experiments/insplad_fullview/run.sh smoke

# 仅训A，不自动运行NCut
bash experiments/insplad_fullview/run.sh train-only

# 对已完成的本协议A5000补充NCut；省略--run则读取LATEST_TRAIN
bash experiments/insplad_fullview/run.sh ncut --run /完整训练运行目录
```

没有自动关机，没有后台常驻监控，没有根据测试分数自动延长训练。
每次训练创建带北京时间与PID的新目录；中断后保留最近已完整写入的备份。
`last.pt`包含优化器，当前入口不提供无缝resume功能；不要把备份存在解释为已实现断点续训。

## 文件结构与反馈

```text
insplad_fullview_B/
  runs/train_seed1_<时间>_<PID>/
    config.json / input_view.json / provenance.json / source/
    dataset_manifest.csv / dataset_counts.json
    initialization_check.json / log.txt / run_status.json
    results/
      config.json / input_view.json
      last.pt
      checkpoints/step_002500.pt / step_005000.pt
      summary_002500.json / summary_005000.json
      metrics_005000.csv / image_scores_005000.csv
    ncut/step_005000_<时间>_<PID>/
      summary.json / metrics.csv / defect_metrics.csv
      image_scores.csv / quality.json / reference_used.json
    training_complete.json
    A_vs_NCut.json / NCUT_RESULTS.txt / REPORT.md
  references/<指纹>/calibration.json / normal_references.csv
  checks/... / smoke/...
  LATEST_TRAIN.txt / LATEST_A_RESULTS.txt / LATEST_NCUT.txt
```

训练完成后提供运行目录中的`results/summary_005000.json`与`A_vs_NCut.json`路径。
预计训练约1～1.5小时，NCut约5～10分钟；以当前机器日志为准。
结果含两份轻量权重和一个覆盖式优化器备份，通常约1～2GiB，无冻结编码器重复副本或大型特征缓存。
