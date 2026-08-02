# RTX 3090 全量离线实验报告

实验日期：2026-08-01。硬件为 NVIDIA RTX 3090 24GB、4 vCPU、30GB RAM，Ubuntu 22.04，
独立 Python 3.10 环境。所有排序指标使用相同的 20,000 用户、`seed=42` 用户级切分、最后一次点击答案，
且分母包含候选池未覆盖答案的用户。召回全量指标使用相同的 200,000 用户最后点击留出集。

原始数据、数千万行特征表、模型 checkpoint 和完整候选表没有提交到 Git；本地同步的 CSV/JSON 指标位于
`artifacts/remote_run/gpu_3090_full_run/metrics/`。

## 1. 结论摘要

- 旧五路 Min-Max 等权 Top50 的验证候选覆盖率仅 `0.30295`，是旧融合显著差于 ItemCF 的直接原因。
- 深召回把 ItemCF/Embedding/UserCF 的通道深度扩到 150/100/100；诊断额外运行 Direct YouTubeDNN Top100，
  RRF Top150 在 200,000 用户上的命中率为 `0.748360`，完整并集为 `0.767960`。
- 深召回的 LightGBM Classifier、LambdaRank 和完整 DIN 均产生了同一候选表上的真实预测；DIN 不是 smoke 指标。
- 0.05 步长验证集搜索选出 Classifier/LambdaRank/DIN=`0.35/0.50/0.15`，NDCG@5=`0.285786`，
  HitRate@5=`0.40945`。相对同候选集两模型 `0.70/0.30/0.00` 融合，NDCG@5 绝对提升 `0.002851`
  （约 `1.01%`），HitRate@5 绝对提升 `0.00445`。
- Hard-negative 20/50 都严重退化，当前保留 `legacy_sampling + all_groups`。这是一条真实负结果。
- Cold Start 最终实验未启用；不能据此宣称其有效。Direct YouTubeDNN 修复后 Recall@50 从旧实现的
  `0.101415` 提高到 `0.108055`，但仍弱，不建议默认进入融合。

## 2. 阶段一：A–E 排序消融

| 实验 | 候选覆盖率 | 模型 | 特征数 | NDCG@5 | HitRate@5 | NDCG@10 | HitRate@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| A 旧五路 Min-Max Top50 | 0.30295 | Classifier | 25 | 0.197792 | 0.26095 | 0.208373 | 0.29305 |
| A 旧五路 Min-Max Top50 | 0.30295 | LambdaRank | 25 | 0.193479 | 0.25615 | 0.204999 | 0.29130 |
| B ItemCF Top50 | 0.62975 | Classifier | 25 | 0.278162 | 0.39650 | 0.314926 | 0.50985 |
| B ItemCF Top50 | 0.62975 | LambdaRank | 25 | 0.276455 | 0.39235 | 0.313283 | 0.50585 |
| C 三路 RRF Top150、旧特征 | 0.65580 | Classifier | 25 | 0.270114 | 0.39035 | 0.311344 | 0.51695 |
| C 三路 RRF Top150、旧特征 | 0.65580 | LambdaRank | 25 | 0.274129 | 0.39355 | 0.313633 | 0.51520 |
| D 三路 RRF Top150、完整特征 | 0.65580 | Classifier | 54 | 0.278748 | 0.39945 | 0.318053 | 0.52025 |
| D 三路 RRF Top150、完整特征 | 0.65580 | LambdaRank | 54 | 0.279241 | 0.39795 | 0.318054 | 0.51735 |
| E 五路 RRF Top150、完整特征 | 0.65740 | Classifier | 54 | 0.278635 | 0.39840 | 0.317649 | 0.51810 |
| E 五路 RRF Top150、完整特征 | 0.65740 | LambdaRank | 54 | 0.279143 | 0.39945 | 0.317164 | 0.51640 |

E 组两模型融合为 Classifier/LambdaRank=`0.6/0.4`，NDCG@5=`0.279946`、HitRate@5=`0.39960`。
五路相对三路只增加 `0.00160` 候选覆盖率，且未稳定改善排序指标，因此 Direct YouTubeDNN 和 Cold Start
保持默认关闭。

## 3. 阶段二：负采样和 LambdaRank group

| 采样 | Group | 训练行数 | 全负用户 | NDCG@5 | HitRate@5 |
|---|---|---:|---:|---:|---:|
| legacy | all | 1,008,197 | 61,401 | 0.279241 | 0.39795 |
| legacy | positive only | 1,008,197 | 61,401 | 0.277389 | 0.39540 |
| hard20 | all | 3,718,599 | 61,401 | 0.029692 | 0.05005 |
| hard20 | positive only | 3,718,599 | 61,401 | 0.027047 | 0.04515 |
| hard50 | all | 9,118,599 | 61,401 | 0.011717 | 0.01665 |
| hard50 | positive only | 9,118,599 | 61,401 | 0.013595 | 0.01840 |

只保留融合排名最靠前的负例造成严重选择偏差，负正比分别升至约 30.35 和 75.89，和验证候选分布不一致。
过滤全零标签 group 也没有改善全用户指标。因此后续使用 legacy 随机采样并保留全部 group。

## 4. 阶段三：深召回

通道深度为 ItemCF Top150、Embedding Top100、YouTubeDNN UserCF Top100；Direct YouTubeDNN Top100
只用于诊断，Cold Start 关闭。RRF 使用 `rrf_k=60`，最终保留 Top150。

| 通道 | 平均候选 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Recall@150 |
|---|---:|---:|---:|---:|---:|---:|
| ItemCF | 141.796 | 0.385005 | 0.500330 | 0.632335 | 0.713600 | 0.746445 |
| Embedding | 75.385 | 0.019960 | 0.062825 | 0.168755 | 0.242660 | 0.242660 |
| YouTubeDNN UserCF | 74.596 | 0.021290 | 0.046860 | 0.174310 | 0.270510 | 0.270510 |
| YouTubeDNN Direct（旧候选诊断） | 99.095 | 0.023640 | 0.044545 | 0.101415 | 0.174730 | 0.174730 |

| 融合指标 | 数值 |
|---|---:|
| RRF@10 | 0.360265 |
| RRF@50 | 0.621570 |
| RRF@100 | 0.706125 |
| RRF@150 | 0.748360 |
| 完整并集命中率 | 0.767960 |
| 平均并集候选数 / 最大值 | 295.554 / 447 |

ItemCF@50 命中 126,467 人；RRF@50 命中 124,314 人，其中保留 ItemCF 命中 121,563、丢失 4,904、
新增 2,751。到 RRF Top150 时，ItemCF@50 命中丢失为 0，并新增 23,205 人。这证实 Top50 不应作为
最终召回池，Top150/200 才能让弱通道的互补候选进入精排。

基于这批候选重新训练的两个 LightGBM 模型融合（Classifier/LambdaRank=`0.7/0.3`）达到
NDCG@5=`0.282935`、HitRate@5=`0.40500`。

## 5. 阶段四：YouTubeDNN 修复和全量训练

修复内容包括：保留 0 为 padding、真实 item ID 全部偏移 1、真实 user ID 映射、FAISS item ID 反向映射、
过滤已点击文章、候选不足补齐、TensorFlow 显存按需增长、OOM batch 回退、训练曲线和最佳权重保存。

- 训练样本 675,899，验证样本 200,000，用户 200,000，item 26,343。
- Batch size 256，1 epoch；训练 16.64 秒，端到端 79.82 秒。
- Recall@10/20/50/100=`0.025205/0.048175/0.108055/0.187735`。
- 每用户严格返回 100 个未点击候选；user embedding 平均范数 1.0。

修复后 Recall@50 比旧候选诊断的 `0.101415` 增加 `0.006640`，但绝对强度仍明显低于 ItemCF，
因此不自动把 Direct 通道权重改为非零。

## 6. 阶段五：DIN 全量训练

- 训练 1,030,260 行，预测 2,961,976 行、20,000 用户。
- 两个 epoch 的验证 AUC 为 `0.916714`、`0.923954`；最佳 checkpoint 已保存。
- 训练 234.79 秒，预测 38.47 秒，诊断函数总耗时 293.51 秒。
- 全用户 NDCG@5=`0.256463`、HitRate@5=`0.37315`；NDCG@10=`0.296868`、HitRate@10=`0.49810`。

DIN 单模弱于 LightGBM，但它和 LightGBM 的用户内排序相关性更低，可能提供融合互补。

## 7. 阶段六：真实预测融合

所有三个文件均为 2,961,976 行、20,000 用户，user-item 键一一匹配，正样本候选覆盖用户 14,943。
先按用户 Min-Max 归一化，再以 0.05 步长搜索 231 个凸权重组合，主目标为全用户 NDCG@5。

| 模型/融合 | Classifier | LambdaRank | DIN | NDCG@5 | HitRate@5 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classifier | 1.00 | 0.00 | 0.00 | 0.280424 | 0.40255 | 0.321649 | 0.52955 |
| LambdaRank | 0.00 | 1.00 | 0.00 | 0.281738 | 0.40345 | 0.321854 | 0.52710 |
| DIN | 0.00 | 0.00 | 1.00 | 0.256463 | 0.37315 | 0.296868 | 0.49810 |
| 原两模型融合 | 0.70 | 0.30 | 0.00 | 0.282935 | 0.40500 | 0.323275 | 0.52940 |
| 最佳三模型融合 | 0.35 | 0.50 | 0.15 | **0.285786** | **0.40945** | **0.326538** | **0.53500** |

平均用户内 Spearman：Classifier–LambdaRank=`0.9781`，Classifier–DIN=`0.7547`，LambdaRank–DIN=`0.7451`。
DIN@5 有 742 个其他两模型均未命中的独占用户，解释了为何弱单模仍可获得 0.15 融合权重。
权重只保存到实验产物，没有自动覆盖默认配置。

## 8. 阶段七：历史长度分组

| 历史组 | 用户数 | Recall@150 | Classifier NDCG/HR@5 | LambdaRank NDCG/HR@5 | DIN NDCG/HR@5 | 融合 NDCG/HR@5 |
|---|---:|---:|---:|---:|---:|---:|
| 1–3 | 12,622 | 0.769371 | 0.290925 / 0.419823 | 0.292167 / 0.419902 | 0.278241 / 0.404690 | **0.296735 / 0.426557** |
| 4–10 | 5,257 | 0.727031 | 0.270974 / 0.381206 | 0.273270 / 0.385391 | 0.233329 / 0.336694 | **0.276350 / 0.391098** |
| >10 | 2,121 | 0.664781 | 0.241351 / 0.352664 | 0.240663 / 0.350306 | 0.184204 / 0.275813 | **0.244014 / 0.353135** |

在完整通道深度下，Embedding 相对 ItemCF 的新增命中在短/中/长历史分别为 85/60/5；UserCF 为
70/66/19；Direct YouTubeDNN 为 136/55/22。Embedding 的 ItemCF 候选重合率也从短历史的 0.633
降到长历史的 0.041，但长历史本身只命中 39 人，说明低重合不等于高价值。当前证据不支持复杂门控模型；
更稳妥的下一步是在独立验证折上测试轻量的历史长度分段权重。Cold Start 本次缺席，只有在新用户、历史极短或
热门/规则召回不足时才应单独启用和评估。

## 9. 实际运行入口

```bash
# 深召回
python run_pipeline.py --mode offline --recall multi \
  --recall-channels itemcf,embedding,youtubednn_usercf \
  --itemcf-topk 150 --embedding-topk 100 \
  --youtubednn-usercf-topk 100 --recall-topk 150 \
  --fusion-method weighted_rrf --rrf-k 60 \
  --experiment-name deep_recall_top150

# YouTubeDNN 诊断
python run_youtubednn_diagnostics.py --data-dir data/raw \
  --output-dir artifacts/offline/youtubednn_full --topk 100

# 独立 DIN 全量训练
python run_din_diagnostics.py --feature-dir artifacts/offline/deep_recall_top150 \
  --output-dir artifacts/offline/din_full

# 三模型预测融合（0.05 步长）
python run_prediction_ensemble_search.py \
  --classifier artifacts/offline/deep_recall_top150/classifier_score_validate.csv \
  --ranker artifacts/offline/deep_recall_top150/ranker_score_validate.csv \
  --din artifacts/offline/din_full/din_score_validate.csv \
  --output-dir artifacts/offline/prediction_ensemble --units 20

# 用户分组
python run_user_group_analysis.py --data-dir data/raw \
  --recall-dir artifacts/offline/deep_recall_top150 \
  --classifier artifacts/offline/deep_recall_top150/classifier_score_validate.csv \
  --ranker artifacts/offline/deep_recall_top150/ranker_score_validate.csv \
  --din artifacts/offline/din_full/din_score_validate.csv \
  --ensemble artifacts/offline/prediction_ensemble/best_ensemble_score_validate.csv \
  --output-dir artifacts/offline/user_groups
```

## 10. 风险和后续建议

1. RRF@50 仍可能低于 ItemCF@50；本次分别是 `0.621570` 和 `0.632335`。如果业务必须直接使用召回
   Top50，可增加 ItemCF Top50 保底配额；若有精排，优先保留 Top150/200 并把选择交给排序模型。
2. RRF@150=`0.748360` 已显著提高覆盖，但完整并集仍有约 1.96 个百分点未进入 Top150，可评估 Top200
   的收益和排序成本。
3. 弱通道要按“新增命中”而非单通道 Recall 决策。Direct YouTubeDNN 和 Cold Start 不应默认等权启用。
4. 已把每路 recalled flag、原始分数、rank、reciprocal rank、RRF 分数、来源数和跨通道一致性接入
   排序特征；后续可增加分段权重，但必须在独立验证集或多折验证上调参。
5. 当前融合权重由同一验证集选择并报告，存在验证集选择偏差；提交线上前应固定权重，并在另一时间窗或折上复核。

## 11. 简历描述与面试讲解

简历描述：基于天池新闻推荐数据构建 ItemCF、内容向量、YouTubeDNN/UserCF 多路召回与 LightGBM/DIN
精排系统；将旧 Min-Max 等权 Top50 融合改为 Weighted RRF 和 Top150 深候选池，并引入 29 个召回来源特征。
在 RTX 3090 上完成 200,000 用户召回与 20,000 用户统一离线验证，候选覆盖率达到 0.74715；三模型融合
NDCG@5/HitRate@5 达到 0.285786/0.40945。实现配置指纹、断点续跑、通道/模型互补诊断、稳定评估和自动化测试。

3–5 分钟讲解提纲：旧系统对每路分数独立 Min-Max 后等权相加，弱通道头部被放大，且 Top50 过早截断，
所以五路覆盖率只有约 0.303。先用完整并集、ItemCF 命中保留和通道新增命中定位问题，再改为只依赖名次的
Weighted RRF，让 ItemCF 权重最高，并把候选池扩到 150。随后把来源 flag/rank/RRF 等 29 个特征交给精排。
负采样实验表明“只取最难负例”会造成分布偏差，因此保留旧随机采样；YouTubeDNN 修复了 padding 和 ID 映射，
DIN 完成全量训练但单模不强。最终用真实预测做用户内归一化和 0.05 权重搜索，DIN 因低相关和独占命中获得
0.15 权重。结果说明工程上不能只看单通道或 AUC，而要统一切分、全用户分母，并同时检查候选覆盖和排序质量。
