# 本次封装验证

验证日期：2026-11-12。使用当前服务器已有两套环境和外部原权重；未复制正式权重或图片到仓库。

## 封存结果重算

Python标准库工具根据三份2656行逐图CSV独立重算AUROC及FPR@95TPR。三套方法具有完全相同的样本集合/标签，A5000基线分数一致；NCut融合算术逐图核对。五类宏平均复现85.33%、88.56%、86.39%。

## 新入口全量推理

| 入口 | 图片数量 | 结果 | 对原逐图分数最大绝对误差 |
|---|---:|---|---:|
| `ncut-evaluate` | 2656 | A85.33%；NCut区域88.28%；NCut融合88.56% | **0.0**，三种评分全部一致 |
| `costfilter-evaluate` | 2656 | A85.33%；CostFilter-only85.13%；融合86.39% | **0.0**，三种评分全部一致 |

新运行输出根目录：`/root/autodl-tmp/experiments/insplad_three_methods_reproduction`。全量summary、日志和分数差异核对报告在`provenance/validation/`。与原代码相比，只处理外部路径、相对清单和检查点路径配置比较；数值方法保持原实验实现。

10张小样本推理也已跑通。小批量改变矩阵运算形状，A/NCut小样本对旧全量分数最大差约1.04e-5；因此小样本只验证执行，不用作正式结果或算法选择。正式32-batch重评估逐图误差为0。

## 训练与协议检查

- A5000入口两步训练检查：初始化RNG和首批采样索引匹配原记录；编码器保持冻结、Bottleneck实际更新；检查点可重新加载；10张评估完成。
- CostFilter两步训练检查通过：A5000无梯度、过滤网络更新、检查点重载输出最大误差0；batch=1/32评估形状均通过。结果在`provenance/validation/costfilter_smoke_training_check.json`。它只验证可执行性，不是5000步正式结果。
- 数据与源文件校验：五类9006正常训练图、2254正常测试图、402异常测试图；50张正常参考哈希通过；A原24文件和CostFilter5个关键运行文件与实验快照一致。
- CostFilter CPU检查：DTD47类5640图、图片完整性和两种合成来源的预览通过；未使用真实异常训练。
- 根`tests/`5项：AUROC并列/反序样例、三方法原结果、空格/中文路径、原源码身份、无权重/图片混入。
- 原NCut5项数值测试、完整视野3项测试通过。
- 原CostFilter8项协议测试通过，包括与原代价体逐元素一致、batch=1、合成范围、worker/恢复确定性、极端概率下有限损失与梯度。

没有重新执行两个5000步完整训练。本次验证了原权重的全量推理复现及两步训练链路；从头训练最终分数仍受GPU及库版本影响，不能将两步检查等同于5000步重新训练确认。

## 可手动重复的检查

```bash
python scripts/verify_results.py --files
python -m unittest discover -s tests -v
bash scripts/run.sh check
bash scripts/run.sh costfilter-check
bash scripts/run.sh ncut-evaluate
bash scripts/run.sh costfilter-evaluate
```

以上两条正式evaluate只推理。训练检查可运行`bash scripts/run.sh a-train --smoke --train-only`及`bash scripts/run.sh costfilter-smoke`，会产生新的诊断权重。不要把smoke结果纳入正式表格。
