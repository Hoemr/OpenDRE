# OOD 实验配置与 Log 指标总结

本文档说明 `dre_ood_train.sh` 能跑出的**所有实验配置**，以及 `training_log.txt` 中的**指标与格式**，便于后续写绘图/制表代码做对比。

---

## 一、脚本能跑出的实验配置

### 1.1 调用方式

```bash
bash dre_ood_train.sh [RUN_MODE] [DATA_MODE] ["GPU_LIST"]
```

- **RUN_MODE**：`neural` | `score` | `all`（默认 `all`）
- **DATA_MODE**：`cifar10` | `cifar100` | `both`（默认 `cifar10`）

### 1.2 方法列表（与 subsub_method 对应）

| 类型 | 方法名（subsub_method） | 说明 |
|------|-------------------------|------|
| **Neural** | `nce`, `pw`, `chisq`, `infonce`, `logistic` | 5 种 neural density ratio，backbone=resnet18 |
| **Score**  | `dre_infty`, `d3re`, `osdre`               | 3 种 score 方法，backbone=unet |

### 1.3 消融维度

- **种子**：`SEEDS=(1 2 3407)`，每个 (方法, 数据集) 会跑 3 个 seed。
- **数据集**：由 DATA_MODE 决定
  - `cifar10` → 只跑 **cifar10**
  - `cifar100` → 只跑 **cifar100**
  - `both` → 先 **cifar10** 再 **cifar100**

**重要**：同一 (dataset, sub_method, subsub_method, seed) 对应**一个实验目录**、一个 `training_log.txt`。

### 1.5 与配置相关的超参（便于制表/筛选）

- **CIFAR-10**：neural/score 均为 `batch_size=512`；CIFAR-100 下 neural=1024，score=768。
- **Neural**：`image_backbone=resnet18`，`image_backbone_norm=groupnorm`。
- **Score**：`image_backbone=unet`，`path_type=trigonometric`，`t_mode=lognorm`，`joint=True`，`bridge`，`condition=True`。
- **OOD 设定**：`ood_use_imglist=True`，`ood_train=random`，`ood_in_dist` = 当前数据集（cifar10 或 cifar100）。

---

## 二、结果目录与 Log 文件位置

### 2.1 目录结构

所有 OOD 实验结果在：

```
results/fdiv_estimation/ood_detection/<ood_in_dist>/
```

其中 `<ood_in_dist>` 为 `cifar10` 或 `cifar100`。

### 2.2 单次实验目录命名规则

每个实验一个子目录，名称由配置拼接而成（含 sub_method、subsub_method、ood_in_dist、backbone、seed 等），例如：

- **Neural**：`dre_neural_<subsub_method>_d3072_<ood_in_dist>_resnet18_random_noise0.005_imglist_seed<seed>/`
- **Score**：`dre_score_<subsub_method>_trigonometric_d3072_<ood_in_dist>_unet_random_noise0.005_imglist_bridge0.01_lognorm_..._cond_joint_seed<seed>/`  
  （osdre 等可能带 `imqrbf_zero_K400` 等后缀）

**建议**：用脚本扫描 `results/fdiv_estimation/ood_detection/{cifar10,cifar100}/` 下所有子目录，再根据目录名或每个目录内的 `training_log.txt` 第一行「Run command」解析出 `sub_method`、`subsub_method`、`ood_in_dist`、`seed`，避免手写复杂正则。

### 2.3 Log 文件

每个实验目录下：

- **`training_log.txt`**：训练与验证的完整日志（见下节）。

---

## 三、training_log.txt 内容结构

### 3.1 开头：运行命令与参数

- 第 1 行：`Run command:` 后跟完整 `python main.py ...` 命令。
- 随后多行：`Run parameters:` 下列出所有运行参数（如 `sub_method`, `subsub_method`, `ood_in_dist`, `seed`, `batch_size`, `image_backbone` 等）。

**解析建议**：从 Run command 或 Run parameters 里提取 `--sub_method`、`--subsub_method`、`--ood_in_dist`、`--seed`，即可唯一标识该实验并用于制表/分组。

### 3.2 模型结构与参数量

- 一段 `Model Summary:`，包含 `OutOfDistributionDetectionModel` 及 backbone（如 `ResNetDensityRatioModel` 或 `ImageJointScoreModel` 等）。
- **参数量**：紧接在 Model Summary 后有 `Trainable params: <N>` 和 `Total params: <N>`（单位：个）。制表时取 **Total params** 即可，可用正则 `Total params:\s*(\d+)` 解析。

### 3.3 训练过程：每 epoch 两行

1. **仅 Loss**（每 epoch 一行）  
   格式：`Epoch <epoch>/1000, Loss <value>`
2. **每 `eval_freq` 个 epoch（默认 10）一次验证**  
   格式一行，包含所有 OOD 指标（见下）。

### 3.4 验证行格式（重要：用于解析指标）

验证行形如（一行，中间用 ` | ` 分隔）：

```
Epoch <N>/1000 | AUROC <v> | FPR95 <v> | AUPR-IN <v> | AUPR-OUT <v> | ACC <v> | near-AUROC <v> | far-AUROC <v> | near-OOD: <name1> <v>, <name2> <v> | far-OOD: <name3> <v>, <name4> <v>
```

**统一指标（每个验证 epoch 都有）**：

| 指标名      | 含义           | 取值   |
|-------------|----------------|--------|
| **AUROC**   | 全体 OOD 平均 AUROC | 0~1   |
| **FPR95**   | TPR=95% 时的 FPR   | 0~1   |
| **AUPR-IN** | In-distribution 的 AUPR | 0~1 |
| **AUPR-OUT**| OOD 的 AUPR        | 0~1   |
| **ACC**     | 分类准确率（ID vs OOD） | 0~1 |
| **near-AUROC** | 近 OOD 平均 AUROC | 0~1 |
| **far-AUROC**  | 远 OOD 平均 AUROC | 0~1 |

**按数据集的 AUROC（在 near-OOD / far-OOD 中）**：

- **ood_in_dist = cifar10**  
  - near-OOD: `cifar10`（val TIN）, `tin`  
  - far-OOD: `mnist`, `svhn`
- **ood_in_dist = cifar100**  
  - near-OOD: `cifar10`, `tin`  
  - far-OOD: `mnist`, `svhn`

解析时可按 `near-OOD:` 和 `far-OOD:` 切分，再按 `name value` 解析每个数据集的 AUROC。

### 3.5 Test 阶段：两行约定与 test_time

训练结束后会跑两次 test（见 `main.py`）：

1. **第一行 Test**：用 **last checkpoint**（最后 epoch 的模型）在测试集上评估。
2. **第二行 Test**：用 **best checkpoint**（val near-AUROC 最高的 ckpt）在测试集上评估。

**制表时统一用第二行**（best_ckpt）的指标。解析时取所有以 `Test |` 开头的行，最后一行即为 best_ckpt 的 test 结果。

Test 行格式与验证行一致，且末尾多一项 **test_time**（本次 test 的推理耗时，单位秒）：

```
Test | AUROC <v> | FPR95 <v> | ... | far-OOD: mnist <v>, svhn <v> | test_time <sec>s
```

- **test_time**：从加载 ID + 所有 OOD 数据到算完所有指标的总时间（单次 test 的 wall-clock）。可用于对比不同方法/backbone 的推理效率。

### 3.6 Early Stopping

- 日志中会出现：`EarlyStopping counter: <k>/30 (best near-AUROC: <best>)`
- **监控指标**：`val_near_auroc`（即验证集的 near-AUROC）
- **Patience**：30 个 eval 周期（即 30×eval_freq 个 epoch，默认 300 epoch）
- 制表/绘图时：验证指标可取 best 验证行；**test 指标取第二条 Test 行（best_ckpt）**；若该 run 未跑 test，则对应 test 指标可填 NaN 或空缺。

### 3.7 单行示例（便于写解析正则）

```
2026-03-13 05:21:06,764 - Epoch 10/1000 | AUROC 0.5994 | FPR95 0.9450 | AUPR-IN 0.1920 | AUPR-OUT 0.9388 | ACC 0.1163 | near-AUROC 0.5285 | far-AUROC 0.6702 | near-OOD: cifar10 0.5080, tin 0.5490 | far-OOD: mnist 0.6886, svhn 0.6519
```

解析建议：

- 用 ` \| ` 分割后，每个 segment 形如 `KEY value` 或 `KEY: name value, ...`。
- 先取 `AUROC`、`FPR95`、`AUPR-IN`、`AUPR-OUT`、`ACC`、`near-AUROC`、`far-AUROC`。
- 再单独解析 `near-OOD:` 和 `far-OOD:` 得到各数据集的 AUROC。

---

## 四、后续绘图与制表可用的维度与指标

- **对比维度**：数据集（cifar10 / cifar100）、方法类型（neural vs score）、具体方法（nce, pw, …, dre_infty, d3re, osdre）、seed（1, 2, 3407）。
- **主要指标**：AUROC、FPR95、near-AUROC、far-AUROC；可选 AUPR-IN、AUPR-OUT、ACC (ACC一般在OOD中用不着)。
- **附加指标**（建议放入表或图注）：
  - **参数量**：从 log 开头 Model Summary 后的 `Total params: <N>` 解析，单位个。
  - **test_time**：从**第二条** Test 行末尾的 `test_time <sec>s` 解析，单位秒（best_ckpt 的整次 test 耗时）。
- **Test 指标取用**：表里统一用**第二行** `Test | ...`（best_ckpt）；第一行为 last_epoch，仅作参考。
- **细粒度**：按 OOD 数据集（cifar10, tin, mnist, svhn 等）的 AUROC 制表或画 bar chart。
- **汇总方式**：同一 (dataset, sub_method, subsub_method) 下 3 个 seed 的 mean±std，再跨方法/类型对比。

若需要，我可以在另一轮对话里直接给出「从 `results/` 扫描 log、解析指标、输出表格/图的 Python 脚本」草稿（含 pandas + matplotlib/seaborn）。
