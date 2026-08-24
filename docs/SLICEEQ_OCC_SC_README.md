# SliceEqOccSC 运行说明

`SliceEqOccSC` 是当前主方法 `SliceEqOcc` 的独立优化版本。它不修改或覆盖父版本源码，默认数据路径、预训练 checkpoint、训练轮数、loss、EMA、batch、验证和推理设置均与父版本一致。

## 为什么值得尝试

当前 SliceEqOcc 对每个中心切片独立采样 `sigma/phase`。这意味着来自同一病例的相邻切片在同一个 volume pass 中可能被赋予互相矛盾的虚拟采集协议。真实 MRI 的 slice profile 和采样网格属于 scan-level protocol，而不是每张切片独立变化。

SliceEqOccSC 将虚拟采集从 `slice-wise IID` 改成 `scan-coherent`：

- 同一病例在一个训练 epoch 内共享同一个连续 `(sigma, phase)`；
- labeled 与 unlabeled 病例分别做随机分层采样，覆盖原来的连续区间；
- 每个 epoch 刷新一次协议，因此长期多样性不下降；
- 仍保留区间 tails，不使用 SliceEqSAQ 的四个固定节点；
- profile 只根据 case identity 和固定 seed 生成，不读取图像、mask、置信度或测试信息。

当前 PROMISE12 H5/list 没有可靠的 spacing/thickness metadata，所以本版本只能称为 **synthetic scan-coherent SliceEqOcc**，不能称为 thickness-calibrated 或 exact scanner PSF。将来若恢复可信 metadata，再实现物理坐标的 protocol-conditioned 阶段。

## 保持不变的设置

```text
root_path:
/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source

pretrained checkpoint:
/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth

max_iterations = 30000
loader batch = 24 = 12 L + 12 U
effective student batch after warmup = 36
sigma = [0.45, 0.85]
phase = [-0.25, 0.25]
protocol refresh = 1 epoch
seed = 1337
```

输出目录与父版本隔离：

```text
../model/SliceEqOccSC_PROMISE12_7_labeled/self_train/unet
```

## 训练

在 `code/` 目录执行：

```bash
python train_sliceeq_occ_sc.py
```

默认第一轮不修改任何参数。特别是不要同时改变 refresh、sigma、phase、loss weight 或 batch，否则无法判断 scan coherence 是否有效。

## 测试

默认会严格读取新实验目录中的：

```text
../model/SliceEqOccSC_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth
```

执行：

```bash
python test_sliceeq_occ_sc.py
```

或者显式冻结一个 checkpoint：

```bash
python test_sliceeq_occ_sc.py \
  --checkpoint_path ../model/SliceEqOccSC_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

## 新增诊断

每 200 步除父版本的 profile/occupancy 日志外，还会输出：

```text
SliceEqOccSC protocol iteration ...:
refresh=...;
unique_case_fraction(L/U)=.../...;
within_case_sigma_range(L/U)=0/0;
within_case_phase_range(L/U)=0/0
```

必须满足：

- warmup 后 `within_case_sigma_range` 和 `within_case_phase_range` 始终为 0；
- profile 和 occupancy 指标保持有限；
- 长期 `sigma/abs_phase/center_w` 仍接近父版本约 `0.65/0.125/0.625`；
- fractional occupancy 不应长期退化为 0；
- 输出目录必须包含 `SliceEqOccSC`，避免覆盖 0.844566 的父版本。

## batch=36 的准确含义

命令行 `--batch_size` 仍然是 24，并没有设置为 36。额外的 12 张是根据同一批 labeled stacks 在 GPU 中构造的 `reacquired-L` view：

```text
12 original-L + 12 reacquired-L + 12 reacquired-U = 36 student views
```

这 12 张对于 fractional occupancy 的数学定义并非必要，但对当前已经得到 0.844566 的完整三分支目标是需要的：

- `original-L` 学习中心切片 hard GT，提供 clean anatomical anchor；
- `reacquired-L` 学习 exact-GT fractional occupancy，建立没有伪标签噪声的 acquisition-equivariance；
- `reacquired-U` 才是半监督 pseudo-occupancy 分支。

若删除这 12 张，会删除 exact-GT acquisition teaching；若用它替换 original-L，则会删除 hard anchor。二者都变成不同方法。显存不足时可以把三组 view 分段 forward/梯度累积，但仍应保留同一总目标、一次 optimizer/EMA update，而不是把方法改成 24-view 后直接比较。

## 本机验证

```bash
python -m py_compile \
  train_sliceeq_occ_sc.py test_sliceeq_occ_sc.py utils/sliceeq_scan.py

python -m unittest \
  ../tests/test_sliceeq_scan.py \
  ../tests/test_sliceeq_scan_contract.py -v
```

当前本机没有 PyTorch，因此 tensor test 需要在 CUDA 服务器执行；语法和 source-contract tests 已在本机执行。

