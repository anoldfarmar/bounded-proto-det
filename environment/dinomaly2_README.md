# 环境记录

原实验使用Python3.11，torch2.9.0+cu128、torchvision0.24.0+cu128，RTX4080SUPER 32GiB。
具体包版本见packages.json。已有AutoDL环境可直接指定PYTHON_BIN，不需要重新安装。

Windows阅读源码、CSV、JSON和运行标准库verify_results.py无需GPU环境。训练/评估入口面向已配置PyTorch的Linux环境；未打包conda环境或安装轮子。

requirements-runtime.txt是当前复核环境的版本记录，不是包含全部传递依赖的离线安装仓库。2D入口不需要Open3D；xFormers缺失会提示但不阻止本实验运行。
