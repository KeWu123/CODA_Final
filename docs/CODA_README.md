# CoDA-MT 桌面 Baseline 运行说明

CoDA-MT 已作为并行入口部署在本 `Baseline` 文件夹中，原始
`train_baseline.py`、`test_baseline.py` 和数据加载代码均未覆盖。

## 文件

- `code/train_coda.py`：CoDA-MT 训练入口。
- `code/test_coda.py`：与训练实验名匹配的测试入口。
- `code/utils/coda.py`：强视图、局部证据损失、软伪目标及损失函数。
- `code/utils/promise12_preflight.py`：只读检查固定数据划分和 HDF5。
- `tests/test_coda.py`：CoDA 数值与梯度测试。
- `tests/test_deployment_contract.py`：原代码哈希、参数和数据划分契约测试。

## 默认设置

新训练和测试入口默认使用：

```text
root_path = E:/Desktop/PROMISE12
exp       = CoDA_MT_PROMISE12
labelnum  = 7
```

其余 Baseline 设置保持不变：2D U-Net、两类、256×256、10k 监督预训练、
30k 自训练、batch 24=12+12、SGD 0.01、EMA 0.99、seed 1337。

当前数据划分已锁定：

```text
train.list        35 cases
train_slices.list 940 slices
first 7 cases     191 labeled slices
val.list          5 cases
test.list         10 cases
```

`promise12_preflight.py` 会核对四份列表的规范化逐行 SHA-256、数量、
前 7 病例与 191 张切片的边界。LF 与 Windows CRLF 被视为相同，但病例
或切片的增删、改名和重排仍会让训练拒绝启动。

## 数据状态

当前 `E:/Desktop/PROMISE12` 已通过数据预检：

```text
955 referenced assets: HDF5
Git LFS pointers:       0
unknown files:          0
```

抽样检查确认训练切片、验证体和测试体均包含 `image`、`label`，图像为
有限 `float32`，标签为二值 `int8`。如果以后再次复制数据，预检仍会
拦截 Git LFS 指针或非 HDF5 文件。

## 验证

在 PowerShell 中：

```powershell
Set-Location E:\Desktop\Baseline
$env:KMP_DUPLICATE_LIB_OK='TRUE'  # 仅当前 Windows 测试进程需要
python -m unittest discover -s tests -v
```

预期：14 项测试全部通过。

训练入口的数据加载回调定义在模块顶层，并通过 `functools.partial` 绑定
seed，因此兼容 Windows 的 `spawn` 多进程启动方式；预训练与自训练仍均使用
`num_workers=4`，随机种子规则仍为 `seed + worker_id`。

## 训练

真实 H5 准备完成后：

```powershell
Set-Location E:\Desktop\Baseline\code
python train_coda.py
```

从 `code` 目录运行时，模型会按 Baseline 原相对路径写入：

```text
E:\Desktop\Baseline\model\CoDA_MT_PROMISE12_7_labeled\
```

也可以显式传参，但不得更换列表或划分：

```powershell
python train_coda.py --root_path E:/Desktop/PROMISE12 --exp CoDA_MT_PROMISE12
```

训练环境仍需安装 Baseline 原有依赖，包括 `tensorboardX`、PyTorch、
torchvision、NumPy、SciPy、scikit-image、h5py 和 tqdm。

## 测试

训练结束后从同一目录运行：

```powershell
python test_coda.py
```

测试入口默认查找：

```text
../model/CoDA_MT_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth
```

## Shared Pre10000 checkpoint update

The current `train_coda.py` no longer reruns supervised pretraining. It strictly loads
the same fixed Baseline Pre10000 checkpoint used by OBA and SliceEq, restores both
`net` and `opt`, logs SHA-256, resets the self-training RNG, and runs only Self30000:

```text
/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

Normal launch remains `python train_coda.py` from `code/`. The
`--pretrained_checkpoint` argument may override a relocated file, but no checkpoint
search or fallback is performed.
