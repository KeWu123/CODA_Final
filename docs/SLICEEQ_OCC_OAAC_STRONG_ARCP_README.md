# SliceEqOcc-OAAC-Strong-ARCP（H7.18）

ARCP 只校准 SliceEq 的三切片权重，不修改最终 Strong 的网络、teacher、EMA、SGD/lr、loss/ramp、
batch36、OAAC scale1.25、验证、每1000 iter周期保存或2-D推理。

它不需要原始 NIfTI/DICOM，也不声称恢复真实 scanner PSF。参考矩阵只从
`train_slices.list` 对应的 image H5 计算，病例内平均后再病例等权平均；不读取 val/test，也不读取
U label。

## 文件

- `code/utils/sliceeq_arcp.py`：参考矩阵、轴向 Gram response 和权重校准；
- `code/analyze_sliceeq_arcp_gate.py`：零训练 H5-only gate；
- `code/train_sliceeq_occ_oaac_strong_arcp.py`：独立7-label训练入口；
- `code/test_sliceeq_occ_oaac_strong_arcp.py`：严格单 checkpoint 测试；
- `tests/test_sliceeq_arcp.py`：数值合同；
- `tests/test_sliceeq_arcp_contract.py`：父方法冻结与评估合同。

## 1. 服务器预检

```bash
cd /home/aiteam/zhengtaoma/CoDA
python -m unittest \
  tests.test_sliceeq_arcp \
  tests.test_sliceeq_arcp_contract -v
```

## 2. 先运行零训练 gate

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u analyze_sliceeq_arcp_gate.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --output_json ../model/SliceEqOccOAACStrongARCP_PROMISE12_7_labeled/analysis/h7_18_arcp_gate.json
```

只有 JSON 的 `decision` 为 `pass` 才进行30k训练。失败时不要调整 epsilon、alpha阈值、中心权重范围或
诊断网格；保留最终 `SliceEqOccOAACStrong`。

## 3. 训练

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u train_sliceeq_occ_oaac_strong_arcp.py
```

启动时会在实验目录写入 `arcp_reference.json`。训练模型目录为：

```text
../model/SliceEqOccOAACStrongARCP_PROMISE12_7_labeled/self_train/unet/
```

验证仍每200 iter执行，`unet_best_model.pth` 规则不变，普通周期权重仍每1000 iter保存。

## 4. 测试指定 checkpoint

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u test_sliceeq_occ_oaac_strong_arcp.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --checkpoint_path ../model/SliceEqOccOAACStrongARCP_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

测试阶段不使用相邻切片、ARCP 或 OAAC，仍为单张2-D切片推理。
