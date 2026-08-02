# 项目架构

项目代码采用 `src` 布局，根目录中的 `Recall.py`、`tezhenggongcheng.py`、
`rank.py` 和 `rank_final.py` 仅作为历史命令的兼容入口。算法实现位于
`src/tianchi_rec/`。

```text
原始点击与文章数据
        │
        ▼
recall/
├── common.py          数据变换与召回评估
├── itemcf.py          ItemCF 相似度和物品召回
├── content.py         FAISS 内容向量近邻
├── youtube_dnn.py     序列样本、双塔训练与向量召回
├── usercf.py          UserCF 与用户向量近邻
├── cold_start.py      新文章规则过滤
└── fusion.py          Weighted RRF、旧分数融合与候选来源元数据
        │
        ▼
features/
├── data.py            数据切分、缓存与召回结果加载
├── candidates.py      候选转换、标签和负采样
├── builder.py         用户—候选文章交叉特征
├── recall_sources.py  召回来源元数据校验与排序特征
└── user.py            用户、热度、设备和偏好特征
        │
        ▼
ranking/
├── lightgbm_models.py LambdaRank 与二分类模型
├── pipeline.py        排序阶段编排和可选 DIN
├── ensemble.py        权重搜索与模型分数融合
├── scores.py          用户内分数归一化
└── submission.py      Top-K 补全、过滤与提交校验
```

## 设计约束

- 纯算法模块不读取项目固定路径，也不在导入时训练模型。
- 数据目录和产物目录统一由 `tianchi_rec.config` 管理。
- TensorFlow、DeepMatch 和 FAISS 在对应算法运行时才加载；ItemCF 模式不再
  因导入召回阶段而强制加载深度学习依赖。
- `_stage.py` 只负责阶段执行顺序、缓存读写和环境变量兼容。
- 根目录兼容入口继续支持原有流水线，便于旧命令和已有产物断点续跑。

## 离线诊断入口

- `run_gpu_content_similarity.py`：使用 GPU FAISS 生成精确内容向量近邻缓存，并记录可复核元数据。
- `run_youtubednn_diagnostics.py`：独立训练/评估 YouTubeDNN，保存曲线、ID/padding 检查和 Recall。
- `run_din_diagnostics.py`：在已构建特征上独立训练完整 DIN，保存 checkpoint、曲线和统一排序指标。
- `run_prediction_ensemble_search.py`：校验三模型候选键一致性，计算用户内相关性、命中重合并搜索融合权重。
- `run_user_group_analysis.py`：按历史长度输出模型指标及通道新增、独占命中和 ItemCF 重合率。

这些脚本只消费已有离线产物，不改变主流水线接口；大型预测和模型文件仍保留在 artifacts 中，不进入 Git。
