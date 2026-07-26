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
             每个用户 Top 50
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
├── PIPELINE.md              # 云端运行与参数说明
└── 推荐系统/                # 本地数据目录，CSV 不提交到 Git
```

运行过程中会自动创建以下目录：

- `result_full_offline/`：线下召回、训练特征、验证结果和融合权重
- `result_full_online/`：全量测试候选、模型及最终提交文件
- `pipeline_logs/`：各阶段日志

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

## 数据准备

从天池新闻推荐赛题页面下载数据，并解压到 `推荐系统/`：

```text
推荐系统/
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

### 完整训练与预测

```bash
python run_pipeline.py --mode all --recall multi --din --gpu 0
```

最终提交文件生成在：

```text
result_full_online/tianchi_news_submission.csv
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
| `--mode` | `all` | `validate`、`final` 或完整流程 `all` |
| `--recall` | `multi` | `itemcf` 或 `multi` |
| `--valid-users` | `20000` | 线下验证用户数 |
| `--recall-topk` | `50` | 精排前每个用户保留的候选数 |
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

- 当前多路召回使用等权融合，可以基于线下指标学习召回权重或设置通道配额。
- 可将每路召回分数、排名和来源作为独立精排特征，避免融合阶段丢失信息。
- 召回覆盖率仍是主要瓶颈，可增加候选数并尝试 Swing、图召回或序列召回。
- 当前 DIN 是候选感知注意力的简化实现，可继续尝试 DIEN、BST 或 SASRec。
- 可以增加时间窗口切分和多折验证，降低单次留出验证的方差。

## 数据与合规说明

本仓库仅提供算法实现，不重新分发天池比赛数据。使用者应遵守数据集及比赛页面的授权条款。项目暂未声明开源许可证；如需复用或分发代码，请先确认仓库所有者添加的许可证。

