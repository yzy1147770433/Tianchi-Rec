# 天池新闻推荐：多路召回与融合排序

## RTX 3090 全量验证结果（2026-08-01）

以下数字均来自同一 `seed=42`、20,000 用户最后一次点击留出验证集，分母包含候选池未覆盖答案的用户；完整记录见
[实验报告](docs/experiment_report.md)。这不是测试集或线上成绩。

| 配置 | 候选覆盖率 | NDCG@5 | HitRate@5 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|---:|
| 深召回 + Classifier | 0.74715 | 0.280424 | 0.40255 | 0.321649 | 0.52955 |
| 深召回 + LambdaRank | 0.74715 | 0.281738 | 0.40345 | 0.321854 | 0.52710 |
| 深召回 + DIN（全量训练） | 0.74715 | 0.256463 | 0.37315 | 0.296868 | 0.49810 |
| 三模型验证集融合（0.35/0.50/0.15） | 0.74715 | **0.285786** | **0.40945** | **0.326538** | **0.53500** |

深召回使用 ItemCF Top150、Embedding Top100、YouTubeDNN UserCF Top100，并以 Weighted RRF 保留
Top150。200,000 用户召回评估中，ItemCF@50 为 `0.632335`，RRF@50 为 `0.621570`，
RRF@100 为 `0.706125`，RRF@150 为 `0.748360`，完整四路候选并集为 `0.767960`。
这说明 RRF@50 仍可能低于强通道，但扩大候选池后覆盖率显著上升，应由精排模型输出最终 Top5。

已完成的 GPU 实验包括：A–E 排序消融、三种负采样与两种 LambdaRank group 策略、深召回、
修复后的 YouTubeDNN 全量训练、DIN 全量训练、0.05 步长三模型融合搜索及历史长度分组分析。
Cold Start 在最终深召回中未启用；相关效果不被表述为已验证结果。

## Recommended v2 defaults

The default multi-recall profile now uses only ItemCF, content Embedding, and
YouTubeDNN UserCF. Weighted RRF uses weights `1.0 / 0.2 / 0.2`, `rrf_k=60`,
and keeps 150 candidates for ranking. Direct YouTubeDNN and Cold Start remain
available for ablation, but their default weights are zero.

```bash
python run_pipeline.py --mode offline --recall multi \
  --recall-profile recommended_v2 \
  --fusion-method weighted_rrf --recall-topk 150 --rrf-k 60 \
  --experiment-name recommended_v2_top150

# Explicit channel selection is also supported.
python run_pipeline.py --mode offline --recall multi \
  --recall-channels itemcf,embedding,youtubednn_usercf \
  --fusion-method weighted_rrf --recall-topk 150 --rrf-k 60 \
  --experiment-name recommended_v2_top150
```

Recall-source metadata is validated and added to the real ranking matrices,
including RRF score, source count, per-channel flag/score/rank/reciprocal rank,
best/mean rank, and cross-channel consistency flags. Missing ranks use 151.
See [pipeline commands](docs/pipeline.md) for isolated experiments, ranking
ablation, cache validation, and the optional DIN smoke test.

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
python run_pipeline.py --mode all --recall multi \
  --recall-profile recommended_v2 --recall-topk 150 \
  --fusion-method weighted_rrf --rrf-k 60 \
  --experiment-name recommended_v2_top150
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
python run_pipeline.py --mode all --recall multi \
  --recall-profile recommended_v2 --recall-topk 150 \
  --experiment-name recommended_v2_top150 --resume
```

只有当召回方式、候选数量、DIN 配置和原始数据均未变化时才使用 `--resume`，否则可能混用过期产物。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `all` | `offline`/`validate`、`final` 或完整流程 `all` |
| `--recall` | `multi` | `itemcf` 或 `multi` |
| `--valid-users` | `20000` | 线下验证用户数 |
| `--recall-profile` | `recommended_v2` | 默认三路配置；也支持 `all_channels`、`itemcf_only` |
| `--recall-channels` | 未指定 | 显式选择通道，优先于 profile |
| `--experiment-name` | `recommended_v2_top150` | 隔离召回、特征、模型和分数产物 |
| `--recall-topk` | `150` | 精排前每个用户保留的融合候选数 |
| `--itemcf-topk` | `50` | ItemCF 单通道召回深度 |
| `--embedding-topk` | `50` | 内容 Embedding 单通道召回深度 |
| `--youtubednn-topk` | `20` | YouTubeDNN Direct 单通道召回深度 |
| `--youtubednn-usercf-topk` | `50` | YouTubeDNN UserCF 单通道召回深度 |
| `--cold-start-topk` | `100` | Cold Start 单通道召回深度（默认通道关闭） |
| `--fusion-method` | `weighted_rrf` | `weighted_rrf` 或旧版 `legacy_score_fusion` |
| `--rrf-k` | `60` | RRF 排名平滑常数 |
| `--channel-weights` | 配置默认值 | JSON 格式的通道权重覆盖 |
| `--recall-only` | 关闭 | 只运行召回、评估与诊断，不运行特征和排序 |
| `--run-ablation` | 关闭 | 保存 A-I 召回消融结果 |
| `--weight-search` | 关闭 | 逐通道搜索 RRF 权重（不覆盖默认配置） |
| `--rank-models` | `classifier` | 可选 `classifier,ranker`；先运行分类模型，再按需运行 LambdaRank |
| `--disable-recall-source-features` | 关闭 | 消融时关闭召回来源特征 |
| `--negative-sampling-strategy` | `legacy_sampling` | `legacy_sampling`、`hard_negative_20` 或 `hard_negative_50` |
| `--hard-negative-random-count` | `0` | 困难负样本之外追加的确定性随机尾部数量 |
| `--ranker-group-policy` | `all_groups` | LambdaRank 保留全部 group 或仅保留含正样本的训练 group |
| `--din` | 关闭 | 是否训练 DIN |
| `--din-batch-size` | `64` | DIN batch size，显存不足时可改为 32 |
| `--din-epochs` | `2` | DIN 训练轮数 |
| `--gpu` | `0` | 使用的 GPU 编号 |
| `--resume` | 关闭 | 已有完整产物时跳过对应阶段 |

## 离线实验结果

当前保存日志对应同一批 20,000 个验证用户、三路 Weighted RRF Top 150。指标分母包含
候选集中没有正样本的用户，因此候选覆盖率是排序指标的真实上限：

| 模型/特征组 | MRR | HitRate@5 | NDCG@5 | HitRate@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Classifier，无来源特征 | 0.255856 | 0.390350 | 0.270114 | 0.516950 | 0.311344 |
| Classifier，54 个完整特征 | 0.263405 | **0.399450** | 0.278748 | **0.520250** | 0.318053 |
| LambdaRank，54 个完整特征 | **0.264290** | 0.397950 | **0.279241** | 0.517350 | **0.318054** |

- 验证候选集真实下一点击覆盖率：`0.655800`（13,116 / 20,000）。
- 完整来源特征相对无来源特征的 Classifier：NDCG@5 `+0.008634`，HitRate@5 `+0.009100`。
- DIN 只完成了 20,000 行训练/20,000 行验证的 1 epoch 环境冒烟测试，不能与上表比较；
  未伪造或外推完整 DIN 指标。
- A/B/E 高成本对照组尚未运行时，`ranking_ablation_results.csv` 会明确记录
  `missing_artifacts`，不会填入虚构结果。

实验结果受数据版本、随机种子、依赖版本和硬件环境影响，表中结果用于说明当前实现，不代表比赛最优成绩。

## 核心方法

### 多路召回

1. **ItemCF**：基于用户点击共现计算文章相似度，并加入点击方向、位置、点击时间、文章发布时间和内容相似度权重。
2. **内容向量召回**：将文章 Embedding 单位化，通过 FAISS 内积搜索得到余弦相似文章。
3. **YouTubeDNN**：使用用户历史序列训练双塔向量，通过 sampled softmax 降低全量文章分类成本。
4. **Embedding UserCF**：使用 YouTubeDNN 用户向量检索相似用户，再扩展相似用户点击过的文章。
5. **冷启动召回**：结合用户历史类别、平均字数和文章新鲜度筛选未出现在点击日志中的文章。

多路融合默认使用 Weighted Reciprocal Rank Fusion：每路先按原始分数降序、去重，
再为候选累计 `channel_weight / (rrf_k + rank)`。默认权重在
`src/tianchi_rec/config.py` 中统一维护：ItemCF 为 `1.0`，Embedding 与 YouTubeDNN
UserCF 均为 `0.20`，YouTubeDNN direct 与 Cold Start 为 `0.0`。因此默认只融合前三路；
后两路仍可显式启用做消融。旧 Min-Max 等权融合仍可通过
`--fusion-method legacy_score_fusion` 运行。

召回阶段会保存兼容旧流水线的 `final_recall_items_dict.pkl`。Weighted RRF 还会保存
`final_recall_candidate_sources.pkl`，其中包含与最终候选顺序对齐的各通道排名，可展开为
`rrf_score`、通道数、来源标记、原始分数、rank、倒数 rank、best/mean rank 和跨通道
一致性特征。元数据会进行配置指纹和候选一一对齐校验，缺失 rank 使用 `topk + 1`。

常用离线命令：

```bash
# ItemCF（与多路融合使用同一逐用户最后点击答案集）
python run_pipeline.py --mode offline --recall itemcf --recall-topk 150 \
  --experiment-name itemcf_top150 --recall-only

# 旧 Min-Max 等权融合
python run_pipeline.py --mode offline --recall multi --recall-profile all_channels \
  --fusion-method legacy_score_fusion --recall-topk 150 \
  --experiment-name legacy_five_top150 --recall-only

# Weighted RRF
python run_pipeline.py --mode offline --recall multi \
  --recall-channels itemcf,embedding,youtubednn_usercf \
  --fusion-method weighted_rrf --recall-topk 150 --rrf-k 60 \
  --experiment-name recommended_v2_top150 --recall-only

# 基于已有五路 pickle 产物运行消融，避免重新训练 YouTubeDNN
python run_recall_experiments.py --experiment ablation --recall-topk 150 --rrf-k 60

# 可选逐通道权重搜索
python run_recall_experiments.py --experiment weight-search --recall-topk 150 --rrf-k 60

# 汇总已有 A-E 排序实验；缺失实验只标记，不伪造结果
python run_ranking_experiments.py --experiment ablation --models classifier,ranker

# 复用同一份 RRF 候选，运行负采样与 LambdaRank group 消融
python run_negative_sampling_experiments.py \
  --base-result-dir artifacts/offline/recommended_v2_top150 \
  --output-dir artifacts/offline/negative_sampling_ablation

# 深召回：各通道独立 TopK，最终仍保留 RRF Top150
python run_pipeline.py --mode offline --recall multi \
  --recall-channels itemcf,embedding,youtubednn_usercf \
  --itemcf-topk 150 --embedding-topk 100 \
  --youtubednn-usercf-topk 100 --recall-topk 150 \
  --experiment-name deep_recall_top150
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

- Weighted RRF 仍不保证 RRF@50 一定高于 ItemCF@50；召回池默认扩大到 150，精排负责最终 Top 5。
- 每路排名和来源元数据已接入 LightGBM 与 DIN 的真实训练/预测矩阵；主召回接口仍保持 `(item_id, score)` 兼容格式。
- 召回覆盖率仍是主要瓶颈，可增加候选数并尝试 Swing、图召回或序列召回。
- 当前 DIN 是候选感知注意力的简化实现，可继续尝试 DIEN、BST 或 SASRec。
- 可以增加时间窗口切分和多折验证，降低单次留出验证的方差。

## 数据与合规说明

本仓库仅提供算法实现，不重新分发天池比赛数据。使用者应遵守数据集及比赛页面的授权条款。项目暂未声明开源许可证；如需复用或分发代码，请先确认仓库所有者添加的许可证。
