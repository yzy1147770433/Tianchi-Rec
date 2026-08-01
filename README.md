# 天池新闻推荐：多路召回与融合排序

一个面向天池新闻推荐数据集的端到端推荐系统。项目采用工业界常见的两阶段架构：先通过多路召回从文章库中生成候选集，再使用 LightGBM 和 DIN 对候选文章精排，最终为每个测试用户输出 Top 5 新闻。

> 数据集、缓存特征和模型权重体积较大，不包含在本仓库中。请按照[数据准备](#数据准备)章节在本地放置数据。

## 项目亮点

- ItemCF、内容向量、YouTubeDNN、用户向量 UserCF、冷启动五路召回
- 使用 FAISS 完成文章和用户向量的近邻检索
- 基于最后一次点击构造时间一致的离线验证集
- 用户、文章、上下文、召回及用户—文章交叉特征
- LightGBM LambdaRank、LightGBM 二分类、简化版 DIN 多模型排序
- 以验证集 NDCG@5 自动搜索模型融合权重
- 线下验证与线上预测产物隔离，支持断点续跑

## 系统流程

```text
点击日志 + 文章属性 + 文章 Embedding
                    │
                    ▼
              按用户时间排序
                    │
              最后一次点击留出
                    │
                    ▼
       ┌──────── 多路召回 ────────┐
       │ ItemCF / 内容向量        │
       │ YouTubeDNN / UserCF      │
       │ 新文章冷启动             │
       └──────────────────────────┘
                    │
          按用户归一化、融合、去重
                    │
             每个用户 Top 200
                    ▼
             用户—文章特征工程
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   LambdaRank   LGBMClassifier   DIN
        └───────────┼───────────┘
                    │
        分数归一化 + NDCG 权重搜索
                    │
          去历史点击 + 热门文章补全
                    ▼
               Top 5 推荐结果
```

## 项目结构

```text
.
├── Baseline.py              # 基础 ItemCF 示例
├── Recall.py                # 多路召回及召回融合
├── tezhenggongcheng.py      # 样本构造与特征工程
├── rank_final.py            # 排序模型、评估、融合与提交生成
├── rank.py                  # 排序阶段兼容入口
├── run_pipeline.py          # 跨平台端到端流水线入口
├── run_pipeline.ps1         # Windows PowerShell 启动脚本
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 包元数据、可选依赖和命令行入口
├── src/tianchi_rec/
│   ├── config.py            # 数据、产物和日志的统一路径配置
│   ├── evaluation/          # HitRate、MRR、NDCG 等排名指标
│   ├── recall/              # ItemCF、内容、UserCF、YouTubeDNN、冷启动与融合
│   ├── features/            # 数据切分、候选标注、候选/用户特征
│   ├── ranking/             # LightGBM、DIN、模型融合与提交构造
│   └── stages.py            # 召回和特征阶段的延迟执行入口
├── tests/                   # 不依赖完整比赛数据的单元测试
├── data/
│   ├── README.md            # 数据下载与放置说明
│   └── raw/                 # 本地原始数据，CSV 不提交到 Git
├── artifacts/               # 模型、特征及提交文件，不提交到 Git
│   ├── offline/             # 线下验证产物
│   └── online/              # 线上预测产物
├── logs/                    # 分阶段运行日志，不提交到 Git
└── docs/
    ├── architecture.md      # 模块职责与依赖关系
    └── pipeline.md          # 云端运行与参数说明
```

运行过程中会自动创建以下目录：

- `artifacts/offline/`：线下召回、训练特征、验证结果和融合权重
- `artifacts/online/`：全量测试候选、模型及最终提交文件
- `logs/`：各阶段日志

这些目录均已写入 `.gitignore`。

## 环境要求

推荐环境：

- Ubuntu 22.04 或 Windows 10/11
- Python 3.10
- 16 核 CPU、64 GB 内存、100 GB 可用磁盘
- 完整 DIN 训练建议使用显存不低于 16 GB 的 NVIDIA GPU

创建环境并安装依赖：

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `deepctr`、`deepmatch` 与 TensorFlow/Keras 的版本兼容性较敏感，请优先使用锁定的 Python 3.10 和依赖版本。

开发模式下可额外安装项目命令行入口和测试依赖：

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## 数据准备

从天池新闻推荐赛题页面下载数据，并解压到 `data/raw/`：

```text
data/raw/
├── train_click_log.csv
├── testA_click_log.csv
├── articles.csv
├── articles_emb.csv
└── sample_submit.csv          # 可选，当前提交代码不依赖该文件
```

流水线启动时会检查前四个必需文件。数据文件不会被 Git 跟踪，请勿将比赛数据直接提交到公开仓库。

## 快速开始

### CPU 冒烟测试

先使用 ItemCF 和较小验证集检查环境：

```bash
python run_pipeline.py --mode validate --recall itemcf --valid-users 2000
```

安装项目后也可以使用统一包入口，参数与原命令完全一致：

```bash
python -m tianchi_rec --mode validate --recall itemcf --valid-users 2000
# 或：tianchi-rec --mode validate --recall itemcf --valid-users 2000
```

### 完整训练与预测

```bash
python run_pipeline.py --mode all --recall multi --din --gpu 0
```

最终提交文件生成在：

```text
artifacts/online/tianchi_news_submission.csv
```

Windows 也可直接运行：

```powershell
.\run_pipeline.ps1
```

### 断点续跑

```bash
python run_pipeline.py --mode all --recall multi --din --gpu 0 --resume
```

只有当召回方式、候选数量、DIN 配置和原始数据均未变化时才使用 `--resume`，否则可能混用过期产物。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `all` | `offline`/`validate`、`final` 或完整流程 `all` |
| `--recall` | `multi` | `itemcf` 或 `multi` |
| `--valid-users` | `20000` | 线下验证用户数 |
| `--recall-topk` | `200` | 精排前每个用户保留的融合候选数 |
| `--fusion-method` | `weighted_rrf` | `weighted_rrf` 或旧版 `legacy_score_fusion` |
| `--rrf-k` | `60` | RRF 排名平滑常数 |
| `--channel-weights` | 配置默认值 | JSON 格式的通道权重覆盖 |
| `--recall-only` | 关闭 | 只运行召回、评估与诊断，不运行特征和排序 |
| `--run-ablation` | 关闭 | 保存 A-I 召回消融结果 |
| `--weight-search` | 关闭 | 逐通道搜索 RRF 权重（不覆盖默认配置） |
| `--din` | 关闭 | 是否训练 DIN |
| `--din-batch-size` | `64` | DIN batch size，显存不足时可改为 32 |
| `--din-epochs` | `2` | DIN 训练轮数 |
| `--gpu` | `0` | 使用的 GPU 编号 |
| `--resume` | 关闭 | 已有完整产物时跳过对应阶段 |

## 离线实验结果

当前保存日志对应 20,000 个验证用户、每用户 50 个候选：

| 模型 | MRR | HitRate@5 | NDCG@5 | HitRate@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| LightGBM LambdaRank | 0.041689 | 0.058400 | 0.035139 | 0.105850 | 0.050359 |
| LightGBM Classifier | **0.128728** | **0.228150** | **0.146269** | **0.286500** | **0.165273** |
| DIN | 0.048591 | 0.061750 | 0.041290 | 0.108650 | 0.056366 |

- 候选集真实下一点击覆盖率：`0.310400`
- NDCG@5 网格搜索选择的融合权重：Classifier `1.0`，LambdaRank `0.0`，DIN `0.0`
- 最终提交规模：50,000 个测试用户，每个用户 5 篇文章

实验结果受数据版本、随机种子、依赖版本和硬件环境影响，表中结果用于说明当前实现，不代表比赛最优成绩。

## 核心方法

### 多路召回

1. **ItemCF**：基于用户点击共现计算文章相似度，并加入点击方向、位置、点击时间、文章发布时间和内容相似度权重。
2. **内容向量召回**：将文章 Embedding 单位化，通过 FAISS 内积搜索得到余弦相似文章。
3. **YouTubeDNN**：使用用户历史序列训练双塔向量，通过 sampled softmax 降低全量文章分类成本。
4. **Embedding UserCF**：使用 YouTubeDNN 用户向量检索相似用户，再扩展相似用户点击过的文章。
5. **冷启动召回**：结合用户历史类别、平均字数和文章新鲜度筛选未出现在点击日志中的文章。

五路融合默认使用 Weighted Reciprocal Rank Fusion：每路先按原始分数降序、去重，
再为候选累计 `channel_weight / (rrf_k + rank)`。默认权重在
`src/tianchi_rec/config.py` 中统一维护：ItemCF 为 `1.0`，Embedding、YouTubeDNN、
YouTubeDNN UserCF 均为 `0.20`，Cold Start 为 `0.05`。旧 Min-Max 等权融合仍可通过
`--fusion-method legacy_score_fusion` 运行。

召回阶段会保存兼容旧流水线的 `final_recall_items_dict.pkl`。Weighted RRF 还会保存
`final_recall_candidate_sources.pkl`，其中包含与最终候选顺序对齐的各通道排名，可展开为
`rrf_score`、通道数、来源标记和各路 rank 特征。

常用离线命令：

```bash
# ItemCF（与多路融合使用同一逐用户最后点击答案集）
python run_pipeline.py --mode offline --recall itemcf --recall-topk 200 --recall-only

# 旧 Min-Max 等权融合
python run_pipeline.py --mode offline --recall multi --fusion-method legacy_score_fusion --recall-topk 200 --recall-only

# Weighted RRF
python run_pipeline.py --mode offline --recall multi --fusion-method weighted_rrf --recall-topk 200 --rrf-k 60 --recall-only

# 基于已有五路 pickle 产物运行消融，避免重新训练 YouTubeDNN
python run_recall_experiments.py --experiment ablation --recall-topk 200 --rrf-k 60

# 可选逐通道权重搜索
python run_recall_experiments.py --experiment weight-search --recall-topk 200 --rrf-k 60
```

### 排序特征

- 候选文章与最近点击文章的内容相似度
- 文章发布时间差、字数差及相似度统计量
- 召回分数与召回名次
- 用户活跃度和平均点击间隔
- 设备、操作系统、国家、地区、来源等上下文
- 用户阅读时间、文章发布时间、字数和类别偏好
- 候选文章类别是否命中用户历史兴趣

### 评估指标

- `Recall Hit Rate`：真实下一次点击是否进入候选集，决定精排效果上限
- `HitRate@K`：真实点击是否出现在最终前 K 位
- `MRR`：真实点击排名倒数的均值
- `NDCG@K`：同时关注是否命中以及命中位置

## 已知限制与后续方向

- Weighted RRF 仍不保证 RRF@50 一定高于 ItemCF@50；召回池默认扩大到 200，精排负责最终 Top 5。
- 可将已保存的每路排名和来源元数据接入精排；当前主接口仍保持 `(item_id, score)` 兼容格式。
- 召回覆盖率仍是主要瓶颈，可增加候选数并尝试 Swing、图召回或序列召回。
- 当前 DIN 是候选感知注意力的简化实现，可继续尝试 DIEN、BST 或 SASRec。
- 可以增加时间窗口切分和多折验证，降低单次留出验证的方差。

## 数据与合规说明

本仓库仅提供算法实现，不重新分发天池比赛数据。使用者应遵守数据集及比赛页面的授权条款。项目暂未声明开源许可证；如需复用或分发代码，请先确认仓库所有者添加的许可证。
