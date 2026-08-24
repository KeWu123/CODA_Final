# SliceEqOccCAP 运行说明

`SliceEqOccCAP` 是从原始 `SliceEqOcc` 独立分出的单变量实验。原始文件没有修改，默认数据路径、预训练 checkpoint、30k 更新、seed、profile 范围、监督分支、EMA、验证与推理均沿用 SliceEqOcc。

唯一变化发生在 1,000 iter 之后的无标注分支：同一个三切片栈、同一个 `sigma` 同时生成 `+phase` 和 `-phase` 两个配对采集视图，二者分别使用匹配的 fractional occupancy，最后平均两个 U loss。

## 训练

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python train_sliceeq_occ_cap.py
```

默认输出目录：

```text
../model/SliceEqOccCAP_PROMISE12_7_labeled/self_train/unet
```

训练后 1,000 iter 的 student forward 是 48 views：

```text
12 original-L + 12 reacquired-L + 12 primary-U + 12 phase-reflected-U
```

loader 的 `--batch_size` 仍必须为 24，`--labeled_bs` 仍为 12。两个 U loss 各乘 0.5，所以总 consistency 权重不变。显存不足时不要自行把 batch 改小；先反馈 GPU 和峰值显存，再实现保持同一目标的分段前向版本。

日志中应看到：

```text
SliceEqOccCAP antithetic iteration ...
```

其中 `phase_residual` 必须恒为 `0`；`weight_l1`、`image` 或 `occupancy` separation 不应长期全部为零。

## 测试

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python test_sliceeq_occ_cap.py \
  --checkpoint_path ../model/SliceEqOccCAP_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

不要用模糊 checkpoint 搜索，也不要测试多个相邻 checkpoint 后挑最高结果。请返回训练 `log.txt` 与测试 `performance.txt`。

## 文件

- `code/train_sliceeq_occ_cap.py`
- `code/test_sliceeq_occ_cap.py`
- `code/utils/sliceeq_antithetic.py`
- `tests/test_sliceeq_antithetic_contract.py`
- `tests/test_sliceeq_antithetic.py`

