# SliceEq 运行说明

`train_sliceeq.py` 是自训练入口，不会重复运行 pretrain。默认直接读取 Baseline
现有实验使用的唯一 Pre10000 权重，并严格恢复其中的 `net` 和 `opt`。代码不会搜索、
排序或自动替换 checkpoint。

## 训练

在训练服务器的仓库 `code/` 目录执行：

```bash
python train_sliceeq.py
```

默认权重为：

```text
/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

如服务器上的文件被移动，仍可用
`--pretrained_checkpoint /absolute/path/to/unet_best_model.pth` 显式覆盖；这只是路径覆盖，
不会触发搜索。

默认值保持当前实验配置：

- 数据：`/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`
- seed：1337
- 7 个 labeled cases / 191 个 labeled slices
- loader batch：24（12 labeled + 12 unlabeled）
- 30k self-training iterations，SGD 0.01，EMA 0.99
- SliceEq：相邻三切片，`sigma=[0.45,0.85]`，`phase=[-0.25,0.25]`
- 输出：`../model/SliceEq_PROMISE12_7_labeled/self_train/unet`

student 的 batch 仍是 24，不需要因为本方法增大 batch。第 1000 步后 teacher 会在
`no_grad` 下处理 12×3 张相邻切片以形成硬 LCC mask stack，所以计算量约增加，但
不会把网络改成 2.5D，也不会增加推理显存或推理开销。

启动时日志必须出现：

```text
Loaded shared pretrain checkpoint: ...
Shared pretrain SHA-256: ...
SliceEq profile: offsets=(-1,0,1), ...
```

若 checkpoint 不存在、缺少 `net`/`opt`、列表中的切片编号不连续，程序会直接停止，
不会静默退回其他权重或中心切片。

## TensorBoard 重点观察

- `info/val_mean_dice`：与 CoDA/OBA 相同的验证曲线。
- `sliceeq/target_changed_fraction`：重采集目标相对中心 teacher mask 的改变比例。
- `sliceeq/image_absolute_change`：重采集图像的实际扰动量。
- `sliceeq/center_weight_mean`：slice profile 的中心权重。
- `sliceeq/center_foreground_fraction` 与
  `sliceeq/reacquired_foreground_fraction`：监控目标体积偏移。
- `sliceeq/neighbor_clamped_sample_fraction`：病例首尾切片的边界复制比例。

前 1000 步 `target_changed_fraction` 和 `image_absolute_change` 按设计均为 0。1000 步后，
如果 `target_changed_fraction` 长期几乎为 0，说明 paired target 实际未激活；如果前景
比例发生异常持续漂移，应立即终止，而不是等 test 后再调参。

## 测试

默认只读取 SliceEq 的精确预期路径，关闭跨实验自动搜索：

```bash
python test_sliceeq.py
```

也可以进一步显式指定最终权重：

```bash
python test_sliceeq.py \
  --checkpoint_path /absolute/path/to/unet_best_model.pth
```

## 本地验证状态

新增源码已经通过 Python 语法编译和 6 项 SliceEq 源码契约测试；全仓库 21 项契约测试
通过（另有 2 项因本机无数据而跳过）。当前 Windows 轻量 Python 环境没有可加载的
PyTorch，张量数值测试应在正式 CUDA 环境运行：

```bash
python -m unittest tests.test_sliceeq -v
```
