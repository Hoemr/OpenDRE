# OOD 任务：training_step / validation_step 的 batch 构成与两种模式关系

## 1. 结论概览

- **训练阶段**：两种模式（built-in 与 imglist）都使用**同一套** `OODDetectionDataset`，batch 形式完全一致，都是 `(px, qx, _) = (ID 样本, OOD 样本, 0.0)`。
- **验证/测试阶段**：`validation_step` / `test_step` 拿到的 batch 都来自 **DummyDataset**，只是占位；真正评测在 **on_validation_epoch_end / on_test_epoch_end** 里用 `in_dist_dataset` + `get_eval_ood_datasets()` 现建 DataLoader 算分。
- 因此：**built-in 可以看成 imglist 的特例**——都是「训练时一个 ID + 一个 OOD 源，评测时多个 OOD 数据集」，只是数据来源不同（torchvision vs 列表文件）。

---

## 2. training_step 的 batch 构成

### 2.1 数据源

两种模式都使用 **同一个类** `OODDetectionDataset`（定义在 `pl_data/ood_detection.py`）：

```python
class OODDetectionDataset(Dataset):
    def __init__(self, in_dist_dataset, ood_dataset, noise_std=0.0):
        self.in_dist_dataset = in_dist_dataset   # ID 数据集
        self.ood_dataset = ood_dataset           # 训练用 OOD 数据集（只有一个）
        ...

    def __getitem__(self, idx):
        x_in, _ = self.in_dist_dataset[idx]
        ood_idx = random.randint(0, len(self.ood_dataset) - 1)
        x_ood, _ = self.ood_dataset[ood_idx]
        # 可选噪声
        return x_in, x_ood, torch.tensor(0.0)
```

也就是说，**每个样本**都是：一条 ID + 一条随机采样的 OOD，再加一个占位标量。

### 2.2 DataLoader 打包后

- `train_dataloader()` 对 `train_dataset`（即 `OODDetectionDataset`）做 `DataLoader(..., batch_size, shuffle=True)`。
- 一个 batch 是多个 `(x_in, x_ood, 0.0)` 的 stack，因此 batch 形状为：
  - `batch[0]`: ID 图像，`(B, C, H, W)`
  - `batch[1]`: OOD 图像，`(B, C, H, W)`
  - `batch[2]`: 标量，未使用

### 2.3 在 base 里的 prepare_batch（OOD 分支）

`pl_models/base.py` 里对非 mi_realdata 的 fdiv 任务：

```python
def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    px, qx, true_mi = batch
    return px, qx, true_mi
```

对应关系：

- **px** = `batch[0]` = **ID 样本**（in-distribution）
- **qx** = `batch[1]` = **OOD 样本**（训练时用的那一个 OOD 源）
- 第三项在 OOD 里未用。

### 2.4 在 ood_detection 的 training_step

```python
def training_step(self, batch, batch_idx):
    px, qx, _ = self.prepare_batch(batch)
    loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
    ...
```

即：**训练时每个 step 的 batch 就是 (ID batch, OOD batch)**，和是否 imglist 无关；区别只在于这两个 dataset 是怎么来的（见下）。

---

## 3. 两种模式下「训练用 OOD」的来源（仅影响 batch 内容来源）

| 项目           | Built-in 模式                         | Imglist 模式                          |
|----------------|----------------------------------------|----------------------------------------|
| ID 数据        | torchvision（如 CIFAR10 train）        | 列表文件（如 `train_cifar10.txt`）     |
| 训练用 OOD     | `_get_ood_dataset(train=True)` 一个源  | 配置里的一个列表（如 `val_tin.txt`）   |
| OOD 源数量     | 1（local / general / universal 三选一）| 1（`ood_train` 对应一个 list）         |
| Batch 结构     | `(px, qx, _)` 同左                    | `(px, qx, _)` 同左                     |

- **Built-in**  
  - `general`：例如 CIFAR-10 时 OOD = SVHN（一个固定数据集）。  
  - `universal`：OOD = `300K_random_images.npy`。  
  - `local`：OOD = 同 ID 的 crop 版本。  
  都是「一个 in_dist_dataset + 一个 ood_dataset」交给 `OODDetectionDataset`。

- **Imglist**  
  - ID = 某个 `id["train"]` 的 ImglistDataset（再包成 ImglistDatasetAsTuple，仍满足 `(data, label)`）。  
  - OOD = `ood_train` 对应的一个 ImglistDataset（如 `val_tin.txt`）。  
  同样是「一个 ID + 一个 OOD」交给同一个 `OODDetectionDataset`。

所以从 **training_step 看到的 batch 构成** 来说，两种模式一致；差异只在「ID / OOD 各自来自 torchvision 还是 imglist」。

---

## 4. validation_step / test_step 的 batch 构成

两个模式里 **val_dataset / test_dataset** 都是 **DummyDataset**：

```python
# ood_detection.py (built-in)
self.val_dataset = DummyDataset(input_shape=(3, 32, 32))
self.test_dataset = DummyDataset(input_shape=(3, 32, 32))

# ood_detection.py (imglist mode)
self.val_dataset = DummyDataset(input_shape=(3, image_size, image_size))
self.test_dataset = DummyDataset(input_shape=(3, image_size, image_size))
```

```python
class DummyDataset(Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return torch.zeros(self.input_shape), 0
```

因此：

- **validation_step(batch, batch_idx)** 收到的 batch 来自 `val_dataloader()`：
  - 只有 1 个样本，batch = `(zeros(1,C,H,W), 0)`，仅占位，**不参与任何真实指标计算**。
- **test_step** 同理，batch 也是 Dummy 的占位。

真正做评测的是 **on_validation_epoch_end / on_test_epoch_end**：

- 用 `datamodule.train_dataset.in_dist_dataset` 得到 ID 数据；
- 用 `datamodule.get_eval_ood_datasets(stage="val"/"test")` 得到**多个** OOD 数据集；
- 在代码里**临时**建 DataLoader，对 ID 和每个 OOD 分别算 score，再算 AUROC/FPR95 等。

也就是说：**传进 validation_step / test_step 的 batch 只是占位；评测逻辑与这两个 step 的 batch 无关，两种模式在这点上也一样。**

---

## 5. 评测阶段：多 OOD 数据集从哪来

- **Built-in**  
  - `get_eval_ood_datasets()` 读 `OOD_EVAL_CONFIG[ood_in_dist]["ood_datasets"]`（如 cifar10 → `['svhn','cifar100','lsun','celeba']`），用 `_load_ood_benchmarks` 按名字用 torchvision 等加载，返回**多个** Dataset（或 ConcatDataset）。  
  - 和训练时用的「一个 OOD 源」可以完全不同（例如训练用 universal，评测用 svhn/cifar100/lsun/celeba）。

- **Imglist**  
  - `get_eval_ood_datasets()` 返回 `self.eval_ood_datasets`，即配置里 `ood_eval` 的多个列表（如 val, cifar100, tin, mnist, svhn）对应的多个 ImglistDataset。  
  - 训练时只用 `ood_train`（一个 list），评测用多个 list。

所以：**训练 = 一个 ID + 一个 OOD；评测 = 同一个 ID + 多个 OOD**，两种模式都是这套规则，只是数据来源不同。

---

## 6. 小结表

| 阶段           | 数据来源（built-in）           | 数据来源（imglist）        | Batch 形式 / 用途                    |
|----------------|--------------------------------|----------------------------|--------------------------------------|
| **训练**       | ID: torchvision；OOD: 一个源   | ID: 一个 list；OOD: 一个 list | 一致：`(px, qx, _)` = (ID, OOD, _)   |
| **验证 step**  | DummyDataset                   | DummyDataset               | 一致：占位，不参与指标                |
| **验证 epoch_end** | in_dist + get_eval_ood_datasets（多个） | 同上                        | 不依赖 step 的 batch，自己建 Loader  |
| **测试**       | 同验证                         | 同验证                     | 同上                                  |

因此可以认为：**built-in OOD 是 imglist 的特例**——同一套「训练用单 OOD、评测用多 OOD」的流程，只是 built-in 用 torchvision + 固定名字的 OOD，imglist 用列表文件 + 配置里的 near/far 等命名。
