# SliceEqSAQ 完整版本运行说明

`SliceEqSAQ` 是 `SliceEqOcc` 的独立后继版本，不会覆盖已有训练代码或模型目录。
它保留现有 fractional occupancy、EMA、loss 和 batch，只将随机 profile 抽样改为
每个 batch 内严格均衡的四节点采集积分。

## 训练

在 `code/` 目录运行：

```bash
python train_sliceeq_saq.py
```

固定默认输入保持不变：

```text
data: /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source
checkpoint: /home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
seed: 1337
loader batch: 24 = 12 L + 12 U
student batch after warmup: 36 = 12 original-L + 12 reacquired-L + 12 reacquired-U
```

新输出目录：

```text
../model/SliceEqSAQ_PROMISE12_7_labeled/self_train/unet
```

它与 `SliceEqOcc_PROMISE12` 完全分离。

## 启动时检查参数

终端开头应出现以下信息：

```text
START SliceEqSAQ SELF-TRAINING
SliceEqSAQ profile domain: offsets=(-1,0,1), sigma=[0.450, 0.850], phase=[-0.250, 0.250]
SliceEqSAQ quadrature nodes (sigma,phase):
(0.534530,-0.144338), (0.534530,0.144338),
(0.765470,-0.144338), (0.765470,0.144338)
SliceEqSAQ node allocation: four nodes x three samples in each 12-sample branch
SliceEqSAQ effective student batch after warmup: 36
SliceEqSAQ permutation RNG seeds: unlabeled=1337, labeled=1338
```

同时检查打印的 `Namespace` 中：

```text
exp='SliceEqSAQ_PROMISE12'
batch_size=24
labeled_bs=12
seed=1337
sliceeq_sigma_min=0.45
sliceeq_sigma_max=0.85
sliceeq_phase_min=-0.25
sliceeq_phase_max=0.25
```

## 训练中检查采样器

每 200 步新增一行：

```text
SliceEqSAQ quadrature iteration ...:
coverage(L/U)=1.000/1.000;
max_count_deviation(L/U)=0.000000/0.000000
```

在前 1000 步 identity warmup 中这两个 coverage 为 0；1000 步以后必须始终是
`1.000/1.000`，count deviation 必须始终为 0。长期 profile 均值应固定接近：

```text
sigma = 0.650000
abs_phase = 0.144338
center_weight = 0.625217
```

如果 1000 步后 coverage 不是 1、count deviation 非 0，或者 batch size 被改得不能
被四整除，应停止训练，不要继续使用该结果。

## 测试

```bash
python test_sliceeq_saq.py
```

测试默认：

```text
--save_result False
--auto_find_checkpoint False
```

需要测指定 checkpoint 时显式传入：

```bash
python test_sliceeq_saq.py \
  --checkpoint_path ../model/SliceEqSAQ_PROMISE12_7_labeled/self_train/unet/iter_23000_dice_xxxx.pth
```

## 验证

```bash
python -m py_compile train_sliceeq_saq.py test_sliceeq_saq.py utils/sliceeq_saq.py
python -m unittest ../tests/test_sliceeq_saq.py ../tests/test_sliceeq_saq_contract.py -v
```
