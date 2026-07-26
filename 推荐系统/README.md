# 数据目录

请将从天池新闻推荐赛题页面下载的 CSV 文件放在此目录：

```text
train_click_log.csv
testA_click_log.csv
articles.csv
articles_emb.csv
sample_submit.csv        # 可选
```

原始比赛数据体积较大，且可能受数据集授权条款约束，因此 `.gitignore` 会忽略本目录中的 CSV 文件。不要使用 `git add -f` 将数据推送到公开仓库。

