# 运行前验证：2026-11-06（北京时间）

正式5000步训练尚未启动。已完成以下必要验证。

- 独立Conda环境：`/root/autodl-tmp/envs/costfilter_ad`；`pip check`通过，项目依赖导入通过。
- 五类数据清单与原完整视野实验SHA256一致：9006正常训练、2254正常测试、402异常测试。
- DTD完整性：47类5640张图片，位于`/root/autodl-tmp/data/dtd/dtd/images`。
- 8项CPU检查通过：正常训练边界、完整视野变换、两类合成源、所有增强器dtype、跨worker/恢复样本一致性、原版代价体数值、原A评估组批、饱和概率下的有限损失和梯度。
- GPU两步验证：batch12，正常前向/反向/更新；A5000被冻结且没有梯度；过滤网络确实更新。
- 输入`12×3×392×392`，代价体`12×768×2×64×64`，输出异常分割两类、资产分类五类。
- 过滤网络保存/严格加载后输出最大差异0。
- 推理batch1和batch32都通过，覆盖正式评估的尾批情况。
- 两步训练显存峰值3.202GiB；包含额外推理形状检查的峰值3.680GiB。不是长训练最终峰值承诺。
- 最新可恢复检查点498.89MiB，最终过滤网络权重166.36MiB。
- 在新环境只读重算原A5000的完整2656张测试图，逐图分数最大绝对差异0，宏平均I-AUROC精确为`0.8532808054567191`。

完整A5000验证先发现改变测试batch和跨类别组批会导致细微CUDA数值差异；现已固定为原来的每类batch32、先正常后异常，逐图核对完全一致。没有按测试AUROC调整过滤算法、融合比例或训练步数。

最新GPU验证目录：

```text
/root/autodl-tmp/experiments/insplad_fullview_costfilter/smoke/smoke_seed1_20261106_140507_10340
```

完整A5000一致性报告：

```text
/root/autodl-tmp/experiments/insplad_fullview_costfilter/validation/baseline_check_20261106_140410.json
```

合成预览位于上述smoke目录的`synthetic_preview/index.html`。

Smoke只用每类首张正常和首张异常共10张图，指标不得解读为正式效果。是否提升，需要用户运行5000步并评估完整测试集后判断。

本次没有下载包、模型或DTD，没有修改原Dinomaly2源码或A5000权重；没有启动后台正式训练。
