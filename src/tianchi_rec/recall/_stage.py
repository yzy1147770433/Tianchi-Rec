import pandas as pd
import numpy as np
from tqdm import tqdm
import os, warnings, pickle
import collections
warnings.filterwarnings('ignore')
from pathlib import Path
from tianchi_rec.config import (
    DATA_DIR,
    DEFAULT_FINAL_RECALL_TOPK,
    DEFAULT_RECALL_CHANNEL_TOPKS,
    DEFAULT_RECALL_FUSION_METHOD,
    DEFAULT_RRF_K,
    ITEMCF_CHANNEL,
    OFFLINE_DIR,
    ONLINE_DIR,
    RECALL_CHANNELS,
    RECALL_EVAL_CUTOFFS,
    env_path,
    recall_channel_weights,
    resolve_recall_channels,
)
from tianchi_rec.evaluation.recall_diagnostics import (
    answer_dict,
    evaluate_recall,
    print_recall_metrics,
    run_recall_ablation,
    save_fusion_diagnostics,
    search_rrf_weights,
)
from tianchi_rec.recall import cold_start as cold_start_algo
from tianchi_rec.recall import common as recall_common
from tianchi_rec.recall import content as content_algo
from tianchi_rec.recall import itemcf as itemcf_algo
from tianchi_rec.recall import usercf as usercf_algo
from tianchi_rec.recall import youtube_dnn as youtube_algo
from tianchi_rec.recall import legacy_score_fusion, weighted_rrf_fusion


np.random.seed(42)

OFFLINE = os.environ.get('OFFLINE_VALIDATION', '1') == '1'
RECALL_METHOD = os.environ.get('RECALL_METHOD', 'itemcf')
if RECALL_METHOD not in {'itemcf', 'multi'}:
    raise ValueError("RECALL_METHOD must be either 'itemcf' or 'multi'.")
ITEMCF_SIM_TOPK = int(os.environ.get('ITEMCF_SIM_TOPK', '100'))
SINGLE_RECALL_TOPK = int(os.environ.get('SINGLE_RECALL_TOPK', '50'))
ITEMCF_RECALL_TOPK = int(os.environ.get(
    'ITEMCF_RECALL_TOPK', str(SINGLE_RECALL_TOPK)
))
EMBEDDING_RECALL_TOPK = int(os.environ.get(
    'EMBEDDING_RECALL_TOPK', str(SINGLE_RECALL_TOPK)
))
YOUTUBEDNN_RECALL_TOPK = int(os.environ.get(
    'YOUTUBEDNN_RECALL_TOPK',
    str(DEFAULT_RECALL_CHANNEL_TOPKS['youtubednn_recall']),
))
YOUTUBEDNN_USERCF_RECALL_TOPK = int(os.environ.get(
    'YOUTUBEDNN_USERCF_RECALL_TOPK', str(SINGLE_RECALL_TOPK)
))
COLD_START_RECALL_TOPK = int(os.environ.get(
    'COLD_START_RECALL_TOPK',
    str(DEFAULT_RECALL_CHANNEL_TOPKS['cold_start_recall']),
))
FINAL_RECALL_TOPK = int(os.environ.get(
    'FINAL_RECALL_TOPK', str(DEFAULT_FINAL_RECALL_TOPK)
))
RECALL_FUSION_METHOD = os.environ.get(
    'RECALL_FUSION_METHOD', DEFAULT_RECALL_FUSION_METHOD
)
if RECALL_FUSION_METHOD not in {'weighted_rrf', 'legacy_score_fusion'}:
    raise ValueError(
        "RECALL_FUSION_METHOD must be 'weighted_rrf' or 'legacy_score_fusion'."
    )
RRF_K = int(os.environ.get('RRF_K', str(DEFAULT_RRF_K)))
RUN_RECALL_ABLATION = os.environ.get('RUN_RECALL_ABLATION', '0') == '1'
RUN_RRF_WEIGHT_SEARCH = os.environ.get('RUN_RRF_WEIGHT_SEARCH', '0') == '1'
PIPELINE_CONFIG_FINGERPRINT = os.environ.get('PIPELINE_CONFIG_FINGERPRINT', '')
ENABLED_RECALL_CHANNELS = (
    (ITEMCF_CHANNEL,)
    if RECALL_METHOD == 'itemcf'
    else resolve_recall_channels()
)
print(f'本次实际启用的召回通道: {ENABLED_RECALL_CHANNELS}')
CHANNEL_TOPKS = {
    'itemcf_sim_itemcf_recall': ITEMCF_RECALL_TOPK,
    'embedding_sim_item_recall': EMBEDDING_RECALL_TOPK,
    'youtubednn_recall': YOUTUBEDNN_RECALL_TOPK,
    'youtubednn_usercf_recall': YOUTUBEDNN_USERCF_RECALL_TOPK,
    'cold_start_recall': COLD_START_RECALL_TOPK,
}
if any(value <= 0 for value in CHANNEL_TOPKS.values()):
    raise ValueError(f'每路召回 TopK 必须为正整数: {CHANNEL_TOPKS}')
print(f'本次各通道召回深度: {CHANNEL_TOPKS}')

data_path = DATA_DIR
default_result_dir = OFFLINE_DIR if OFFLINE else ONLINE_DIR
save_path = env_path('RECALL_RESULT_DIR', default_result_dir)
save_path.mkdir(parents=True, exist_ok=True)
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = OFFLINE


# 读取数据
# debug模式： 从训练集中划出一部分数据来调试代码

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集

#################################################################################################
# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(Path(data_path) / 'articles.csv')
    
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df

# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(Path(data_path) / 'articles_emb.csv')
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols], dtype=np.float32)
    # 进行归一化
    item_emb_norm = np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    item_emb_np = item_emb_np / np.maximum(item_emb_norm, 1e-12)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path / 'item_content_emb.pkl', 'wb'))
    
    return item_emb_dict

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# 采样数据
# all_click_df = get_all_click_sample(data_path)

# 全量训练集
all_click_df = recall_common.load_clicks(data_path, offline=OFFLINE)

# 对时间戳进行归一化,用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
item_info_df = get_item_info_df(data_path)
item_content_emb_path = save_path / 'item_content_emb.pkl'
if item_content_emb_path.exists():
    print(f'复用缓存: {item_content_emb_path}')
    item_emb_dict = None  # 后续流程不直接使用该字典
else:
    item_emb_dict = get_item_emb_dict(data_path)


#################################################################################################
# 以下是工具函数
# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}

# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。

# 获取当前数据的历史点击和最后一次点击

# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用



# 获取近期点击最多的文章





#################################################################################################
# 定义多路召回字典
# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = recall_common.item_metadata(item_info_df)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict = {channel_name: {} for channel_name in RECALL_CHANNELS}


# 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
# 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df
    trn_last_click_df = None


#################################################################################################
# 召回效果评估函数
# 固定同一答案用户集作为分母；候选不足某个 K 时自然按实际候选评估。
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5, name='recall'):
    del topk  # 保留旧调用签名；评估点统一由配置定义。
    answers = answer_dict(trn_last_click_df)
    metrics = evaluate_recall(
        user_recall_items_dict,
        answers,
        cutoffs=RECALL_EVAL_CUTOFFS,
        users=answers.keys(),
    )
    print_recall_metrics(name, metrics)
    return metrics


#################################################################################################

# 计算相似性矩阵



# ItemCF 依赖当前抽样用户，必须与本次样本同步生成，不能沿用其他样本的缓存。
i2i_sim = itemcf_algo.item_similarity(trn_hist_click_df, item_created_time_dict)
pickle.dump(i2i_sim, open(save_path / 'itemcf_i2i_sim.pkl', 'wb'))




# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
# user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
# u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item

emb_i2i_sim_path = save_path / 'emb_i2i_sim.pkl'
if emb_i2i_sim_path.exists():
    print(f'复用缓存: {emb_i2i_sim_path}')
    emb_i2i_sim = None  # 在实际召回前再从缓存加载
else:
    item_emb_df = pd.read_csv(data_path / 'articles_emb.csv')
    emb_i2i_sim = content_algo.embedding_similarity(item_emb_df, topk=10)
    pickle.dump(emb_i2i_sim, open(save_path / 'emb_i2i_sim.pkl', 'wb'))
    del item_emb_df




#####################################################################################################
# 多路召回
# 2026/7/3 YoutubeDNN召回
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量

# 将输入的数据进行padding，使得序列特征的长度都一致


# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
need_youtube_embeddings = any(
    channel in ENABLED_RECALL_CHANNELS
    for channel in ('youtubednn_recall', 'youtubednn_usercf_recall')
)
if RECALL_METHOD == 'multi' and need_youtube_embeddings:
    if not metric_recall:
        youtube_recall = youtube_algo.train_youtube_dnn_recall(
            all_click_df, save_path, topk=YOUTUBEDNN_RECALL_TOPK,
        )
    else:
        trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
        youtube_recall = youtube_algo.train_youtube_dnn_recall(
            trn_hist_click_df, save_path, topk=YOUTUBEDNN_RECALL_TOPK,
        )
    if 'youtubednn_recall' in ENABLED_RECALL_CHANNELS:
        user_multi_recall_dict['youtubednn_recall'] = youtube_recall
    if metric_recall and 'youtubednn_recall' in ENABLED_RECALL_CHANNELS:
        metrics_recall(
            user_multi_recall_dict['youtubednn_recall'],
            trn_last_click_df,
            topk=YOUTUBEDNN_RECALL_TOPK,
            name='YouTubeDNN recall',
        )




# 2026/7/4
# 基于商品的召回i2i

# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path / 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = ITEMCF_SIM_TOPK
recall_item_num = ITEMCF_RECALL_TOPK
item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf_algo.recommend_items(
        user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
        item_topk_click, item_created_time_dict, emb_i2i_sim,
    )

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path / 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(
        user_multi_recall_dict['itemcf_sim_itemcf_recall'],
        trn_last_click_df,
        topk=recall_item_num,
        name='ItemCF recall',
    )

if RECALL_METHOD == 'itemcf':
    print(f'ItemCF recall saved to: {save_path / "itemcf_recall_dict.pkl"}')
    raise SystemExit(0)

if 'embedding_sim_item_recall' in ENABLED_RECALL_CHANNELS:
    # 这里是为了召回评估，所以提取最后一次点击
    if metric_recall:
        trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
    else:
        trn_hist_click_df = all_click_df

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
    i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))

    sim_item_topk = ITEMCF_SIM_TOPK
    recall_item_num = EMBEDDING_RECALL_TOPK
    item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)
    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = itemcf_algo.recommend_items(
            user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
            item_topk_click, item_created_time_dict, emb_i2i_sim,
        )
    user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
    pickle.dump(
        user_multi_recall_dict['embedding_sim_item_recall'],
        open(save_path / 'embedding_sim_item_recall.pkl', 'wb'),
    )
    if metric_recall:
        metrics_recall(
            user_multi_recall_dict['embedding_sim_item_recall'],
            trn_last_click_df,
            topk=recall_item_num,
            name='Embedding recall',
        )





# 2026/7/5
#
# 基于用户的召回 u2u2i


# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
# if metric_recall:
#     trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
# else:
#     trn_hist_click_df = all_click_df
    
# user_recall_items_dict = collections.defaultdict(dict)
# user_item_time_dict = get_user_item_time(trn_hist_click_df)

# u2u_sim = pickle.load(open(save_path / 'usercf_u2u_sim.pkl', 'rb'))

# sim_user_topk = 20
# recall_item_num = 10
# item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
#                                                         recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)    

# pickle.dump(user_recall_items_dict, open(save_path / 'usercf_u2u2i_recall.pkl', 'wb'))

# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)


# 使用Embedding的方式获取u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user

# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
if 'youtubednn_usercf_recall' in ENABLED_RECALL_CHANNELS:
    user_emb_dict = pickle.load(open(save_path / 'user_youtube_emb.pkl', 'rb'))
    u2u_sim = usercf_algo.embedding_user_similarity(user_emb_dict, topk=10)
    pickle.dump(u2u_sim, open(save_path / 'youtube_u2u_sim.pkl', 'wb'))

    if metric_recall:
        trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
    else:
        trn_hist_click_df = all_click_df
    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
    u2u_sim = pickle.load(open(save_path / 'youtube_u2u_sim.pkl', 'rb'))
    sim_user_topk = 20
    recall_item_num = YOUTUBEDNN_USERCF_RECALL_TOPK
    item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)
    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = usercf_algo.recommend_from_users(
            user, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
            item_topk_click, item_created_time_dict, emb_i2i_sim,
        )
    user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
    pickle.dump(
        user_multi_recall_dict['youtubednn_usercf_recall'],
        open(save_path / 'youtubednn_usercf_recall.pkl', 'wb'),
    )
    if metric_recall:
        metrics_recall(
            user_multi_recall_dict['youtubednn_usercf_recall'],
            trn_last_click_df,
            topk=recall_item_num,
            name='YouTubeDNN UserCF recall',
        )



#################################################################################################
# 冷启动问题



if 'cold_start_recall' in ENABLED_RECALL_CHANNELS:
    # 先进行 embedding i2i 召回，再进行冷启动规则筛选。
    if metric_recall:
        trn_hist_click_df,_ = recall_common.split_history_last(all_click_df)
    else:
        trn_hist_click_df = all_click_df
    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
    i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))
    sim_item_topk = max(150, COLD_START_RECALL_TOPK)
    recall_item_num = COLD_START_RECALL_TOPK
    item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)
    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = itemcf_algo.recommend_items(
            user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
            item_topk_click, item_created_time_dict, emb_i2i_sim,
        )
    pickle.dump(
        user_recall_items_dict,
        open(save_path / 'cold_start_items_raw_dict.pkl', 'wb'),
    )


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果




    all_click_df_ = trn_hist_click_df.copy()
    all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
    user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = recall_common.user_history_metadata(all_click_df_)
    click_article_ids_set = set(trn_hist_click_df['click_article_id'])
    cold_start_user_items_dict = cold_start_algo.filter_cold_start_items(
        user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict,
        user_last_item_created_time_dict, item_type_dict, item_words_dict,
        item_created_time_dict, click_article_ids_set, recall_item_num,
    )
    pickle.dump(
        cold_start_user_items_dict,
        open(save_path / 'cold_start_user_items_dict.pkl', 'wb'),
    )
    user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict
    if metric_recall:
        metrics_recall(
            user_multi_recall_dict['cold_start_recall'],
            trn_last_click_df,
            topk=FINAL_RECALL_TOPK,
            name='Cold-start recall',
        )



#################################################################################################

# 多路召回合并：默认 Weighted RRF，旧 Min-Max 等权方式保留为消融选项。
channel_weights = recall_channel_weights()
enabled_recall_results = {
    name: user_multi_recall_dict[name]
    for name in ENABLED_RECALL_CHANNELS
}
empty_enabled = [
    name for name, results in enabled_recall_results.items() if not results
]
if empty_enabled:
    raise RuntimeError(f'启用的召回通道没有生成结果: {empty_enabled}')
for channel_name, weight in channel_weights.items():
    if weight > 0 and channel_name not in ENABLED_RECALL_CHANNELS:
        print(f'警告：通道 {channel_name} 未启用，配置权重 {weight} 将被忽略。')
missing_weights = [
    name for name in ENABLED_RECALL_CHANNELS if name not in channel_weights
]
if missing_weights:
    raise ValueError(f'启用通道缺少权重配置: {missing_weights}')
enabled_weights = {
    name: channel_weights[name] for name in ENABLED_RECALL_CHANNELS
}
print(
    f'多路召回合并: method={RECALL_FUSION_METHOD}, topk={FINAL_RECALL_TOPK}, '
    f'rrf_k={RRF_K}, enabled_channels={ENABLED_RECALL_CHANNELS}, '
    f'weights={enabled_weights}'
)
if RECALL_FUSION_METHOD == 'weighted_rrf':
    final_recall_items_dict, recall_source_metadata = weighted_rrf_fusion(
        enabled_recall_results,
        channel_weights=enabled_weights,
        topk=FINAL_RECALL_TOPK,
        rrf_k=RRF_K,
        return_metadata=True,
        itemcf_channel=ITEMCF_CHANNEL,
    )
    recall_source_metadata['config_fingerprint'] = PIPELINE_CONFIG_FINGERPRINT
    with (save_path / 'final_recall_candidate_sources.pkl').open('wb') as file:
        pickle.dump(recall_source_metadata, file, protocol=pickle.HIGHEST_PROTOCOL)
else:
    final_recall_items_dict = legacy_score_fusion(
        enabled_recall_results,
        channel_weights={name: 1.0 for name in enabled_recall_results},
        topk=FINAL_RECALL_TOPK,
    )
    # 覆盖可能残留的 RRF 来源文件，避免下游把旧元数据与新候选错配。
    with (save_path / 'final_recall_candidate_sources.pkl').open('wb') as file:
        pickle.dump(
            {
                'format_version': 1,
                'fusion_method': 'legacy_score_fusion',
                'channel_names': tuple(enabled_recall_results),
                'config_fingerprint': PIPELINE_CONFIG_FINGERPRINT,
                'users': {},
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

with (save_path / 'final_recall_items_dict.pkl').open('wb') as file:
    pickle.dump(final_recall_items_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

if metric_recall:
    metrics_recall(
        final_recall_items_dict,
        trn_last_click_df,
        topk=FINAL_RECALL_TOPK,
        name='Enabled-channel fused recall',
    )
    offline_answers = answer_dict(trn_last_click_df)
    save_fusion_diagnostics(
        enabled_recall_results,
        final_recall_items_dict,
        offline_answers,
        save_path,
        users=offline_answers.keys(),
    )
    if RUN_RECALL_ABLATION:
        run_recall_ablation(
            enabled_recall_results,
            enabled_weights,
            offline_answers,
            save_path / 'recall_ablation_results.csv',
            topk=FINAL_RECALL_TOPK,
            rrf_k=RRF_K,
            users=offline_answers.keys(),
        )
    if RUN_RRF_WEIGHT_SEARCH:
        search_rrf_weights(
            enabled_recall_results,
            offline_answers,
            save_path / 'rrf_weight_search_results.csv',
            topk=FINAL_RECALL_TOPK,
            rrf_k=RRF_K,
            users=offline_answers.keys(),
        )
