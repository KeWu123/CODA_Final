# SliceEqOcc-OAAC-Strong-MPD-PARS

## 方法定位

PARS 是最终 `SliceEqOcc-OAAC-Strong-MPD` 的独立采样 successor。它不增加网络层、不修改 loss、EMA、
学习率、batch 数量或推理，只改变 iter1000 后训练切片进入既定 acquisition risk 的概率。iter0--999
完全复用父 sampler。

当前父 sampler 在 L/U slice indices 中抽样，因此长病例贡献更多更新。PARS 在 L/U stream 中分别先
均匀轮换病人，再从 exact-L 训练前设计并冻结的 three-index-third 分布中选轴向区域，最后在该病例区域
内均匀选 slice。U 分支只使用文件名中的病例与相对 slice index，不读取 U label。

它不是 loss/confidence/uncertainty hard mining；论文贡献边界详见
`research/experiments/h7_slice_profile_reacquisition/h7_20_patient_axial_acquisition_risk_protocol.md`。

## 唯一变化

```text
MPD parent:
  uniform slice-index stream sampling

PARS:
  iter0--999: exact parent sampler
  iter1000+: patient-uniform cycle
      -> frozen exact-L-designed axial-third q
      -> uniform slice within patient-third
```

以下全部不变：MPD q、OAAC Strong1.25、Pre10000、seed1337、SGD/LR0.01、EMA train-mode、teacher
hard+2D-LCC、soft CE+Dice、ramp、12L+12U、student36、30k、validation、每1000 iter保存和2D inference。

## 训练前测试

在训练服务器、`CoDA` 根目录执行：

```bash
python -m unittest tests.test_sliceeq_pars tests.test_sliceeq_pars_contract -v
```

本机没有 NumPy/PyTorch，因此 tensor/numerical suite 必须在训练服务器运行通过。

## 完整训练

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u train_sliceeq_occ_oaac_strong_mpd_pars.py
```

默认使用已锁定路径：

- PROMISE12 H5：`/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`
- shared 7-label Pre10000：父入口中的固定 bundle
- 输出：`../model/SliceEqOccOAACStrongMPDPARS_PROMISE12_7_labeled/self_train/unet`

启动阶段将依次写入：

- `mpd_profile_design.json`
- `pars_sampling_design.json`
- `log.txt`

若任一设计约束、hash、数据边界、CUDA smoke 或 sampler contract 失败，训练会在 iter0 前终止。

## 指定 checkpoint 测试

例如测试 iter29000：

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u test_sliceeq_occ_oaac_strong_mpd_pars.py \
  --checkpoint_path ../model/SliceEqOccOAACStrongMPDPARS_PROMISE12_7_labeled/self_train/unet/iter_29000.pth \
  --auto_find_checkpoint False \
  --save_result False
```

将 `iter_29000.pth` 替换为需要评估的每1000 iter checkpoint 即可。测试入口仍使用原 strict 2-D权重
加载和指标实现。

## 关键日志

启动日志应包含：

- `PARS frozen axial law: parent=...; designed=...`
- `PARS design diagnostics: worst exposure ... -> ...`
- `PARS smoke passed`
- `PARS replaces only TwoStreamBatchSampler`

训练期第1个及每25个 sampler epoch 会记录 L/U three-index-third counts 与 patient count range；平衡轮换下
同一 stream 的 patient count range 应仅相差0或1。
