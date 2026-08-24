# SliceEqOcc-OAAC-Strong：PROMISE12 11-label 运行说明

该入口只把最终方法的标注预算从前7例/191 slices改为前11例/306 slices。以下内容保持不变：U-Net、30k self-training、batch `24=12L+12U`、warmup、SGD、LR、EMA、consistency ramp、SliceEq profile、OAAC Strong scale1.25、验证和测试规则，以及每1000 iteration的普通周期权重保存。

原7-label入口保持不变。11-label入口为：

- `code/pretrain_promise12_label11.py`：可选的10k监督预训练入口；
- `code/train_sliceeq_occ_oaac_strong_label11.py`：最终方法的11-label自训练入口；
- `code/test_sliceeq_occ_oaac_strong.py`：复用严格测试入口，显式传`--labelnum 11`。

## 1. 必须使用11-label预训练

不能把7-label checkpoint用于11-label实验。自训练checkpoint必须是使用前11个训练病例产生的字典，并同时包含：

```text
net
opt
```

若训练服务器上已有符合条件的checkpoint，可跳过预训练。否则在`CoDA/code`中运行：

```bash
cd /home/aiteam/zhengtaoma/CoDA/code

python -u pretrain_promise12_label11.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source
```

默认输出：

```text
/home/aiteam/zhengtaoma/CoDA/model/
SharedPretrain_PROMISE12_11_labeled/pre_train/unet/unet_best_model.pth
```

日志会输出该checkpoint的SHA-256。

## 2. 运行最终11-label方法

```bash
cd /home/aiteam/zhengtaoma/CoDA/code

python -u train_sliceeq_occ_oaac_strong_label11.py \
  --pretrained_checkpoint /home/aiteam/zhengtaoma/CoDA/model/SharedPretrain_PROMISE12_11_labeled/pre_train/unet/unet_best_model.pth
```

默认输出目录：

```text
/home/aiteam/zhengtaoma/CoDA/model/
SliceEqOccOAACStrong_PROMISE12_11_labeled/self_train/unet/
```

训练前会拒绝以下问题：

- 前306 slices不能精确对应`train.list`前11例；
- 11-label病例跨越306 slice边界；
- 未显式提供预训练checkpoint；
- checkpoint缺少`net`或`opt`；
- 任一最终方法参数偏离锁定配置。

## 3. 测试验证集选择的best权重

```bash
cd /home/aiteam/zhengtaoma/CoDA/code

python -u test_sliceeq_occ_oaac_strong.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --exp SliceEqOccOAACStrong_PROMISE12 \
  --labelnum 11 \
  --checkpoint_path /home/aiteam/zhengtaoma/CoDA/model/SliceEqOccOAACStrong_PROMISE12_11_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

指定周期权重时，只替换`--checkpoint_path`，例如：

```text
.../SliceEqOccOAACStrong_PROMISE12_11_labeled/self_train/unet/iter_24000.pth
```

论文主结果仍应按预先确定的验证集best规则选择；1000-step周期权重用于诊断，不应通过test逐个搜索最优iteration。
