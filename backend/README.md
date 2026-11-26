# backend 代码说明

> 所有命令都在**项目根目录**执行，形如 `python -X utf8 -m backend.training.train_d1 ...`。

---

## 1. 数据流

```text
data/<session>/phase2_r<N>/label<L>_pi<P>.pcap       原始 pcap（只读；session = d1 / d3 / …）
data/<session>/session_meta.json                      采集脚本写的网格、设备坐标、每轮记录
        │  python -X utf8 -m backend.data.load_real_stage2 <session> [--sc-from d1]
        ▼
datasets/raw/<session>/static_raw64.npz               未预处理的 64 子载波幅度 / 相位 + 每帧时间戳（主用）
datasets/raw/<session>/static.npz                     已按解析时选定的子载波切好的版本（prepare 的 subcarrier_mode="dataset" 用；D1 起统一用 raw64 + "train_fold"）
datasets/raw/<session>/truncation_report.json         四链路对齐时每点截断了多少帧
        │  backend/training/prepare.py  或  backend/training/crossday.py
        │  （按轮次划分 → 只用训练集拟合预处理器 → 16 帧一个窗口）
        ▼
X : [窗口, 链路 4, 通道 2（幅度 / 相位）, 时间 16, 子载波 55]     y : 15 类 label（label = y×5 + x）
        │  训练 / 评估脚本（第 3 节）
        ▼
backend/artifacts/d1/<select|final>/cls_ampphase_pp2_seed<N>/   D1 模型与 results.json 等
backend/artifacts/crossday/d3/                                  D+3 模型与结果
```

`datasets/raw/` 里还有旧数据 `session_001`、`session_002`（阶段一单链路）与 `session_phase2_001`（10 点 / 1.0 m pilot），只作历史保留；`datasets/manifests/` 是旧 session 的数据清单。

---

## 2. 模块说明

### 2.1 根

| 文件 | 作用 |
|---|---|
| `config.py` | 中央配置：路径、链路顺序（`all_link_ids()`）、预处理参数，以及从 `CSI_PIA_HOST`、`CSI_PIB_HOST`、`CSI_APX_BSSID`、`CSI_APY_BSSID` 读取的设备配置。里面还留有阶段一和旧 pilot 的常量，新代码不要用它推断网格，网格一律走 `sessions.session_grid()` |
| `__init__.py` | 只设置 `LOKY_MAX_CPU_COUNT=1`（避免 Windows 上 joblib 探测 CPU 的噪声） |

### 2.2 `data/`：从 pcap 到数组

| 文件 | 作用 |
|---|---|
| `sessions.py` | **session 的唯一登记处**：原始数据在哪、网格多大（从 `session_meta.json` 读）、各轮次用途（`SESSION_ROUND_ROLES`：train / val / calib / test / perturb）、哪些点被跳过。所有读 pcap 的代码都必须经过这里 |
| `load_real_stage2.py` | 四链路 pcap 解析：每台 Pi 一份 pcap，按 802.11 帧内 BSSID 拆成两条链路，四条按最短截断对齐，输出 `static_raw64.npz`。新 session 用 `--sc-from d1` 沿用 D1 的子载波选择 |
| `load_real.py` | 阶段一单链路 pcap 解析；现在只为 `load_real_stage2` 提供 Nexmon 包格式常量（`CSI_LEN`、`CSI_OFF`、`RAW_SUB`） |
| `load.py` | 读 `datasets/raw/<session>/` 下的 npz（`load_static`、`load_static_raw64`） |
| `preprocess.py` | 预处理（pp2）：只保留 55 个有效子载波并按物理频率排序；幅度 Hampel 去离群 → S-G 平滑 → 标准化；相位线性去趋势 → 标准化。统计量**只用训练集拟合**（`Preprocessor.fit / transform / save / load`） |
| `split.py` | 划分工具（分层随机 / 按批次隔离），被 `prepare.py` 调用 |

### 2.3 `models/`：模型

| 文件 | 作用 |
|---|---|
| `fusion_gat.py` | 提供两样现在还在用的东西：`CNNEncoder`（单链路 2D-CNN 编码器，`freq_bins=16` 时只对时间平均、保留子载波轴，输出 64 维）和 `_TorchReg`（通用训练器：Adam、早停看验证集定位误差、`fit / predict / save / load`）。文件里的 `GATLayer`、`FusionGAT` 和开头文档说的"GAT 主力模型、卡尔曼跟踪"**已不再使用** |
| `fusion_baseline.py` | 论文用的神经网络：`SingleLinkCNNModel`（单链路 CNN，阶段二）、`ConcatMLPModel`（共享编码器 + MLP 头，端到端）、`ConcatMLPMaskModel`（再加训练时随机遮挡链路 p=0.25）、`ConcatMLPSepModel`（4 个不共享编码器，E1 用）。`MeanFusion` 未在 D1 上使用 |
| `knn.py` | KNN（k=5、距离加权、可选训练集标准化），单链路 / 四链路 KNN 与 F-KNN 共用 |
| `sample_gcn.py` | 样本图 GCN（结构移植自前人 Wang / Sun 的 RSSI+GNN）：每个样本一个节点、连 k=5 个最近邻、两层图卷积；推理时测试样本只接收训练邻居的信息（防泄漏）。`window_features()` 也在这里 |
| `metrics.py` | 统一指标：平均误差、P90、准确率、每点准确率、误差 > 1.6 m 比例等。**必须显式传 `grid=`** |

### 2.4 `features/`：链路健康分（预登记负结果）

| 文件 | 作用 |
|---|---|
| `link_health.py` | 从原始 CSI 算每条链路、每个窗口的健康分：①相位残差 ②时间 MAD ③丢包 / 抖动 ④指纹偏离度；`crossday.py` 直接调用 |
| `eval_health.py` | 健康分开发期检验（窗口内一致比例 = AUROC），在 D1 上运行 |
| `gating_check_draft.py` | 开发期端到端门控诊断；用于复现预登记中的开发数据 1.234 → 1.111 m |

### 2.5 `training/`：训练与实验脚本（论文用到的全部在这里）

| 脚本 | 论文位置 | 作用 | 主要产物 |
|---|---|---|---|
| `prepare.py` | — | 把 load → 划分 → 预处理 → 切窗口串起来（`load_stage2_windows`，`mode="by_batch"` 按轮次划分） | — |
| `train_d1.py` | 阶段二、三 | D1 两阶段协议：`select`（训练 r1–r5、r6 早停、看 r7）→ `final`（训练 r1–r8、固定轮数、测 r9 / r10）。模型：单链路 KNN / CNN、四链路 KNN / GCN / ConcatMLP / ConcatMLP-mask；`--model ConcatMLP-sep` 单独跑 | `artifacts/d1/<phase>/cls_ampphase_pp2_seed<N>/results.json` 与模型文件 |
| `random_split_check.py` | 阶段一（本文数据上的直接证据） | D1：按前人方式 6:2:2 随机划分，与按轮次划分对比（2026-10-28，诊断） | `artifacts/d1/random_split/random_split_check.json` |
| `link_drop_check.py` | 阶段三问题② | 推理时去掉一条链路，看依赖哪几条 | `artifacts/d1/final/link_drop_check.json` |
| `crossday.py` | 阶段四 4.1、4.2 | 跨日检验（严格按 `report/健康分预登记.md`，会校验其 md5）：`train --day d3` / `evaluate --day d3 --unblind` / `report`；含"不标定"对照 S4 | `artifacts/crossday/d3/`、`crossday/report.json` |
| `frozen_fusion.py` | 阶段三控制实验 | D1：KNN / GCN / MLP 吃同一份冻结的单链路 CNN 特征 | `…/frozen_fusion.json`、`final/frozen_fusion_summary.json` |
| `frozen_fusion_d3.py` | 阶段四 4.4 | D+3：F-MLP vs ConcatMLP-mask 的验证性检验 | `crossday/d3/frozen_fusion_results.json` |
| `frozen_fusion_nocal.py` | 阶段四 4.1 / 4.4 | D+3 跨天不标定：解耦编码器能否替代标定 | `crossday/d3/frozen_fusion_nocal_results.json` |
| `e1_shared_encoder.py` | 阶段四 4.5 | E1 消融汇总（不共享权重 vs 单独训练） | `final/e1_shared_encoder.json` |
| `e2_linear_probe.py` | 阶段四 4.5 | E2 逐链路线性探测 | `final/e2_linear_probe.json` |
| `e4_redundancy.py` | 阶段四 4.5 | E4 误差多样性（负结果） | `final/e4_redundancy.json` |
| `e5_feature_vs_head.py` | 阶段四 4.5 | E5 表征 / 融合头分解 | `final/e5_feature_vs_head.json` |

每个实验脚本的文件开头都写着**跑之前定下的判据**和运行命令。

---

## 3. 常用命令

```powershell
python -X utf8 -m backend.data.load_real_stage2 d1                     # 解析 D1
python -X utf8 -m backend.data.load_real_stage2 d3 --sc-from d1        # 解析新的一天
python -X utf8 -m backend.training.train_d1 --phase select --task cls --seed 42 1 2
python -X utf8 -m backend.training.train_d1 --phase final  --task cls --seed 42 1 2
python -X utf8 -m backend.training.train_d1 --phase final --summary
python -X utf8 -m backend.training.crossday train --day d3
python -X utf8 -m pytest -q                                            # 测试：tests/test_metrics.py、test_sessions.py
```

---

## 4. 约定与易错点

- **网格**：米制误差必须用 `sessions.session_grid(session)`，不读全局 config。
- **划分**：按轮次（batch）划分，预处理统计量只用训练集；统计单位是轮次。
- **新 session**：采完要在 `sessions.py` 的 `SESSION_ROUND_ROLES` 登记轮次用途，否则全部轮次都当测试。
- **随机性**：GPU 训练不完全确定，同一种子重跑可能差 0.0x m；结果以保存的 json 为准。
- **不要动** `report/健康分预登记.md`（`crossday.py` 会校验 md5）。

---

## 5. 实时预测接口

实时服务已经实现，完整字段见 [`API_CONTRACT.md`](API_CONTRACT.md)。服务从两台 Pi 读取四条链路，以 16 帧窗口运行 F-MLP，并提供三个 HTTP 端点：

- `GET /api/meta`
- `GET /api/status`
- `GET /api/predict/latest`

本版本不提供 WebSocket；`/ws/predictions` 返回 501。客户端约每 0.5 秒轮询状态和最新结果，并用 `session_id` 与 `seq` 去重。

当天 session 已有 `backend/artifacts/live/<session>/train_meta.json` 时可直接加载模型；没有时，服务等待指定标定轮的 15 个格点采完，然后自动解析、训练并进入 `ready`：

```powershell
$env:CSI_PIA_HOST = "<piA-address>"
$env:CSI_PIB_HOST = "<piB-address>"
python -X utf8 -m backend.live.server --session <session> --round 1
```

离线检查可把一轮已采 pcap 送入相同实时缓冲和推理路径：

```powershell
python -X utf8 -m backend.live.server --session <session> --replay data\<session>\phase2_r1
```

任一所需链路超过缓冲时限后，状态变为 `degraded`，新预测暂停。恢复四条链路后才继续产生新 `seq`。旧结果可能仍可从 latest 端点读取，因此调用方还必须检查序号、时间戳和状态。
