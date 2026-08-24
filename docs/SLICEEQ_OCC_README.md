# SliceEqOcc（下一优化版本）运行说明

`SliceEqOcc` 是 SliceEq v1 的独立后继版本。它不会覆盖或修改现有
`train_sliceeq.py`、`test_sliceeq.py`、`utils/sliceeq.py` 与
`dataloaders/sliceeq_dataset.py`。

## 方法变化

- 保留前 1000 步 baseline identity warmup。
- 保留原始 labeled central slice 的 hard CE+Dice anchor。
- 增加一个由真实相邻 GT 共同重采集的 labeled view。
- unlabeled 仍使用 EMA argmax + 2D LCC 的相邻三层 pseudo mask。
- 图像和 mask occupancy 使用同一个 profile；训练直接保留 fractional occupancy，
  不再先做 hard argmax。
- warmup 后 student effective batch 为 36：
  `12 original-L + 12 reacquired-L + 12 reacquired-U`。
- 网络、EMA、优化器、验证和推理图不变。

## 默认训练

在 `code/` 目录执行：

```bash
python train_sliceeq_occ.py
```

默认数据与 checkpoint 路径沿用当前固定设置：

```text
/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source
/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

输出目录与 v1 分离：

```text
../model/SliceEqOcc_PROMISE12_7_labeled/self_train/unet
```

如果显存不足，不要修改 loader batch 或 labeled 比例后直接与 v1 比较；应实现等价的
分段 forward/gradient accumulation，并保持损失归一化和一次 optimizer/EMA update。

## 默认测试

```bash
python test_sliceeq_occ.py
```

测试默认设置：

```text
--save_result False
--auto_find_checkpoint False
```

`--save_results False` 也作为同一参数的兼容别名。即使不保存 NIfTI，测试仍会在精确
snapshot 目录写入 `performance.txt`。如确实需要预测文件，显式执行：

```bash
python test_sliceeq_occ.py --save_results True
```

## 必须保留的诊断

每 200 步的文本日志会包含两行：

```text
SliceEqOcc train iteration ...:
lambda=... loss(original/L-eq/U-eq)=.../.../...;
L-profile(sigma/abs_phase/center_w)=.../.../...; U-profile=.../.../...

SliceEqOcc occupancy iteration ...:
L(frac/entropy/dev/hard_change)=.../.../.../...;
U=.../.../.../...; clamp(L/U)=.../...
```

这些值用于判断 fractional occupancy 是否真正生效，不应从最终版删除。TensorBoard
还包含 labeled/unlabeled 的 profile、occupancy、foreground 和 endpoint-clamping
诊断。

### 终端合理性检查

- 0--1000 步为 identity warmup，因此 fractional/entropy/dev/hard_change 应全为 0，
  `L-eq` 和 `U-eq` loss 也为 0。
- 1000 步后 `sigma` 的长期均值应接近 0.65，`abs_phase` 接近 0.125，
  `center_w` 接近 0.625；单个 batch 会自然波动。
- 所有 `frac/entropy/dev/hard_change/clamp` 必须在 `[0,1]`，且应保持有限值。
- `hard_change` 接近 0 是允许且预期的；新版本依靠 fractional occupancy，不要求
  hard argmax 频繁改变。
- 正常情况下 `entropy <= frac`、`dev <= frac`、`hard_change <= frac`。允许浮点级误差。
- 若 1200 步以后连续多个日志点的 L/U `frac` 都严格为 0，应停止训练并检查邻层索引、
  label stack 或 profile 是否失活。
- 若 U 的 `frac/dev` 长期远高于 L（例如数量级差异），更可能是相邻 pseudo mask
  不稳定，而不是正常 partial volume。
- `clamp` 只应反映病例首尾切片；长期接近 1 表示邻层表或采样范围异常。

## 验证命令

```bash
python -m py_compile \
  train_sliceeq_occ.py test_sliceeq_occ.py utils/sliceeq_occ.py
python -m unittest \
  ../tests/test_sliceeq_occ.py \
  ../tests/test_sliceeq_occ_contract.py -v
```

当前 Windows 工作区已通过语法、关键静态检查以及 23 项跨版本源码契约测试；本地
PyTorch 因 `c10.dll` 初始化失败，tensor 单元测试需在实际 CUDA 训练环境再次执行。
